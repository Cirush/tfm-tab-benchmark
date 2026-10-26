"""BERT para MIDI → tablatura como "rellenar huecos" (Fretting-Transformer §4.4).

Paper: *"The BERT architecture, with a configuration similar to that described
in [20], is trained using masked language modeling. Here, the TAB<#,#> tokens
directly follow the NOTE_ON<#> tokens in the input and are replaced by a <MASK>
token during training."* [20] = MIDI-to-Tab (Edwards et al. 2024): *"hyperparameter
settings that simply halve the bert-base configuration: 384-dimensional hidden
size, 6 hidden layers, 6 attention heads, 1536-dimensional intermediate size,
dropout probability of 10%"*; *"we only compute the loss for string token
predictions"*.

Decisiones:
- Secuencia por nota: NOTE_ON<p> TAB<s,f> TIME_SHIFT<d> NOTE_OFF<p> (encoding v3
  con el TAB tras el NOTE_ON). Entrenamiento: todos los TAB → <MASK>; la loss solo
  cuenta esas posiciones.
- Contexto (protocolo del paper en inferencia: 20 notas + 1 de contexto): la nota
  de contexto lleva su TAB ya decidido, visible. Para que el modelo aprenda a
  usarlo, en el 50% de las ventanas de train el primer TAB queda visible (sin loss).
- Predicción NO autorregresiva: una pasada por trozo, un TAB por nota (no hay
  desalineamiento). Se restringe el argmax a los tokens TAB.
- Ventanas de ≤512 tokens (127 notas a 4 tokens/nota; 512 = máximo de posiciones
  de BERT y la longitud del paper). Mismo optimizador (Adafactor) que T5 y GPT2
  para comparar formulaciones y no optimizadores (MIDI-to-Tab usa AdamW).
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertForMaskedLM

from ..data.chunking import PAPER_CONTEXT, PAPER_INFERENCE_CHUNK, chunk_plan
from ..fretboard import Fretboard
from ..postprocess import match_window, postprocess, raw_alignment
from ..tokenizer_v3 import BOS, EOS, V3Tokenizer
from . import t5_fretting
from .constrained import valid_tab_ids

BERT_CONFIG = dict(hidden_size=384, num_hidden_layers=6, num_attention_heads=6,
                   intermediate_size=1536, hidden_dropout_prob=0.1,
                   attention_probs_dropout_prob=0.1, max_position_embeddings=512)
TOKENS_PER_NOTE = 4


def mask_id(tok: V3Tokenizer) -> int:
    """Id de <MASK> (se añade tras el vocabulario v3)."""
    return tok.vocab_size


def build_model(tok: V3Tokenizer, **overrides) -> BertForMaskedLM:
    cfg = BertConfig(vocab_size=tok.vocab_size + 1, pad_token_id=tok.pad_id,
                     **{**BERT_CONFIG, **overrides})
    return BertForMaskedLM(cfg)


def load_model(path: str) -> BertForMaskedLM:
    return BertForMaskedLM.from_pretrained(path)


def encode(tok: V3Tokenizer, notes, tabs, n_visible: int = 0) -> Tuple[List[int], List[int]]:
    """(ids, labels) de una secuencia. Los TAB de las primeras `n_visible` notas
    se muestran (contexto conocido); el resto se enmascaran y son los que se
    predicen. `tabs` puede ser None para las notas a predecir (inferencia)."""
    m = mask_id(tok)
    ids, labels = [tok.vocab[BOS]], [-100]
    for k, (pitch, dur) in enumerate(notes):
        on, ts, off = tok.to_ids(tok.encode_input([(pitch, dur)]), add_special=False)
        tab = tok.vocab[f"TAB<{tabs[k].string},{tabs[k].fret}>"] if tabs and tabs[k] else -100
        if k < n_visible:
            ids += [on, tab, ts, off]
            labels += [-100] * 4
        else:
            ids += [on, m, ts, off]
            labels += [-100, tab, -100, -100]
    return ids + [tok.vocab[EOS]], labels + [-100]


class WindowDataset(Dataset):
    """Misma interfaz que T5/GPT2: collate -> (input_ids, attention_mask, labels)."""

    def __init__(self, windows: Sequence[dict], tok: V3Tokenizer, seed: int = 42):
        rng = random.Random(seed)
        self.items = [encode(tok, w["notes"], [t for t, _ in w["tabs"]],
                             n_visible=int(rng.random() < 0.5)) for w in windows]
        self.pad = tok.pad_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def collate(self, batch):
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
            labels[i, :len(y)] = torch.tensor(y)
        return ids, mask, labels


@torch.no_grad()
def predict_tracks(
    model, tok: V3Tokenizer, tracks: Sequence[Sequence[Tuple[int, int]]],
    fb: Fretboard = None, chunk: int = PAPER_INFERENCE_CHUNK, ctx: int = PAPER_CONTEXT,
    commit: Optional[int] = None, num_beams: int = 1, batch_size: int = 128,
    constrained: bool = False,
) -> List[Dict[str, list]]:
    """Mismo protocolo y salida que T5/GPT2 (3 etapas). `num_beams` no aplica
    (predicción en una sola pasada). `constrained=True`: en cada hueco solo se
    permiten los TAB que producen la nota (decodificación restringida por pitch)."""
    fb = fb or Fretboard()
    model.eval()
    device = next(model.parameters()).device
    tab_ids = torch.tensor(sorted(i for t, i in tok.vocab.items() if t.startswith("TAB<")))
    not_tab = torch.ones(model.config.vocab_size, dtype=torch.bool)
    not_tab[tab_ids] = False
    not_tab = not_tab.to(device)
    out = [{"raw": [], "window": [], "final": []} for _ in tracks]
    plans = [chunk_plan(len(t), chunk, ctx, commit) for t in tracks]

    for k in range(max((len(p) for p in plans), default=0)):
        active = [i for i, p in enumerate(plans) if len(p) > k]
        for b in range(0, len(active), batch_size):
            ids = active[b:b + batch_size]
            seqs, targets, keep, n_ctx = [], [], [], []
            for i in ids:
                cs, st, end, ce = plans[i][k]
                known = [out[i]["final"][q] for q in range(cs, st)]
                seq, _ = encode(tok, tracks[i][cs:end], known + [None] * (end - st),
                                n_visible=st - cs)
                seqs.append(seq)
                targets.append(tracks[i][st:end])
                keep.append(ce - st)
                n_ctx.append(st - cs)
            L = max(len(x) for x in seqs)
            src = torch.full((len(ids), L), tok.pad_id, dtype=torch.long)
            att = torch.zeros_like(src)
            for j, x in enumerate(seqs):
                src[j, :len(x)] = torch.tensor(x)
                att[j, :len(x)] = 1
            logits = model(input_ids=src.to(device), attention_mask=att.to(device)).logits
            logits = logits.masked_fill(not_tab, float("-inf")).cpu()
            for j, i in enumerate(ids):
                # el TAB de la nota q está en la posición 1 + 4·q + 1
                slots = [2 + TOKENS_PER_NOTE * (n_ctx[j] + q) for q in range(len(targets[j]))]
                rows = logits[j, slots]
                if constrained:
                    allowed = torch.full_like(rows, float("-inf"))
                    for r, (pitch, _) in enumerate(targets[j]):
                        ok = valid_tab_ids(tok, fb, pitch)
                        allowed[r, ok] = 0.0
                    rows = rows + allowed
                pred = tok.decode_target(tok.to_tokens(rows.argmax(-1).tolist()))
                pitches = [p for p, _ in targets[j]]
                n = keep[j]
                out[i]["raw"] += raw_alignment(pred, len(pitches))[:n]
                out[i]["window"] += match_window(pred, pitches, fb)[:n]
                out[i]["final"] += postprocess(pred, pitches, fb)[:n]
    return out


def evaluate_tracks(model, tok: V3Tokenizer, songs: Sequence[dict], **kw) -> Dict[str, float]:
    return t5_fretting.evaluate_tracks(model, tok, songs, predict=predict_tracks, **kw)
