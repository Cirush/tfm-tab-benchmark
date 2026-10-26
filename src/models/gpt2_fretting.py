"""GPT2 para MIDI → tablatura como "completar texto" (Fretting-Transformer §4.4).

Paper: *"the GPT2 model is used. The model is trained to generate tablature
tokens for a given sequence of MIDI tokens in a text completion manner.
Therefore, a 'MIDI:' token followed by the MIDI notes and a 'TABS:' token is
used as a primer sequence. The GPT2 model now completes the text by adding the
tab token sequence."*

Decisiones (el paper no da la configuración del GPT2):
- Tamaño equivalente al T5 del paper: d=128, d_ff=1024, 4 cabezas y 6 capas
  (las 3 del encoder + 3 del decoder del T5) → mismo orden de parámetros, para
  que la comparación sea entre formulaciones y no entre tamaños.
- Mismos datos, troceo, vocabulario, optimizador (Adafactor) y protocolo de
  inferencia que el T5 (20 notas + 1 de contexto, post-procesado ±5).
- La loss solo cuenta los tokens de tablatura (el prompt MIDI no se predice).
- Secuencia: MIDI: <notas> TABS: <TAB/TIME_SHIFT> <eos>. Con contexto, los TABs
  ya decididos de la nota de contexto van tras TABS: como prefijo forzado.
- Generación por lotes con relleno a la izquierda (los prompts tienen longitud
  distinta y GPT2 genera a continuación del último token).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from ..data.chunking import PAPER_CONTEXT, PAPER_INFERENCE_CHUNK, chunk_plan
from ..fretboard import Fretboard
from ..postprocess import match_window, postprocess, raw_alignment
from ..tokenizer_v3 import EOS, V3Tokenizer
from . import t5_fretting
from .constrained import make_prefix_fn

GPT2_CONFIG = dict(n_embd=128, n_inner=1024, n_layer=6, n_head=4, n_positions=1024)


def special_ids(tok: V3Tokenizer) -> Tuple[int, int]:
    """Ids de los tokens 'MIDI:' y 'TABS:' (se añaden tras el vocabulario v3)."""
    return tok.vocab_size, tok.vocab_size + 1


def build_model(tok: V3Tokenizer, **overrides) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=tok.vocab_size + 2,
        bos_token_id=special_ids(tok)[0],
        eos_token_id=tok.vocab[EOS],
        pad_token_id=tok.pad_id,
        **{**GPT2_CONFIG, **overrides},
    )
    return GPT2LMHeadModel(cfg)


def load_model(path: str) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(path)


def prompt_ids(tok: V3Tokenizer, notes, prefix=()) -> List[int]:
    """MIDI: <notas> TABS: [<TABs ya decididos del contexto>]"""
    midi, tabs = special_ids(tok)
    return ([midi] + tok.to_ids(tok.encode_input(notes), add_special=False) + [tabs]
            + tok.to_ids(tok.encode_target(prefix), add_special=False))


def encode_pair(tok: V3Tokenizer, song: dict) -> Tuple[List[int], List[int]]:
    """(secuencia completa, labels): -100 en el prompt, la tablatura + <eos> se predicen."""
    prompt = prompt_ids(tok, song["notes"])
    target = tok.to_ids(tok.encode_target(song["tabs"]), add_special=False) + [tok.vocab[EOS]]
    return prompt + target, [-100] * len(prompt) + target


class WindowDataset(Dataset):
    """Misma interfaz que la del T5: collate -> (input_ids, attention_mask, labels)."""

    def __init__(self, windows: Sequence[dict], tok: V3Tokenizer):
        self.items = [encode_pair(tok, w) for w in windows]
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
    """Mismo protocolo y salida que `t5_fretting.predict_tracks` (3 etapas;
    `constrained=True`: decodificación restringida por pitch)."""
    fb = fb or Fretboard()
    model.eval()
    device = next(model.parameters()).device
    eos = tok.vocab[EOS]
    out = [{"raw": [], "window": [], "final": []} for _ in tracks]
    plans = [chunk_plan(len(t), chunk, ctx, commit) for t in tracks]

    for k in range(max((len(p) for p in plans), default=0)):
        active = [i for i, p in enumerate(plans) if len(p) > k]
        for b in range(0, len(active), batch_size):
            ids = active[b:b + batch_size]
            prompts, targets, keep = [], [], []
            for i in ids:
                cs, st, end, ce = plans[i][k]
                prefix = [(out[i]["final"][q], tracks[i][q][1]) for q in range(cs, st)]
                prompts.append(prompt_ids(tok, tracks[i][cs:end], prefix))
                targets.append(tracks[i][st:end])
                keep.append(ce - st)
            L = max(len(p) for p in prompts)
            src = torch.full((len(ids), L), tok.pad_id, dtype=torch.long)
            mask = torch.zeros_like(src)
            for j, p in enumerate(prompts):           # relleno a la izquierda
                src[j, L - len(p):] = torch.tensor(p)
                mask[j, L - len(p):] = 1
            extra = {}
            if constrained:
                extra["prefix_allowed_tokens_fn"] = make_prefix_fn(tok, fb, targets, L)
            gen = model.generate(
                input_ids=src.to(device), attention_mask=mask.to(device),
                max_new_tokens=2 * chunk + 4, do_sample=False, num_beams=num_beams,
                pad_token_id=tok.pad_id, **extra,
            )
            for j, i in enumerate(ids):
                new = t5_fretting._cut_eos(gen[j, L:].tolist(), eos)
                pred = tok.decode_target(tok.to_tokens(new))
                pitches = [p for p, _ in targets[j]]
                n = keep[j]
                out[i]["raw"] += raw_alignment(pred, len(pitches))[:n]
                out[i]["window"] += match_window(pred, pitches, fb)[:n]
                out[i]["final"] += postprocess(pred, pitches, fb)[:n]
    return out


def evaluate_tracks(model, tok: V3Tokenizer, songs: Sequence[dict], **kw) -> Dict[str, float]:
    return t5_fretting.evaluate_tracks(model, tok, songs, predict=predict_tracks, **kw)
