"""Fretting-Transformer (T5) — modelo, datos y predicción (arXiv:2506.14223).

- Config del paper: d_model=128, d_ff=1024, 3 capas encoder + 3 decoder,
  4 cabezas. d_kv no se especifica -> se deja el valor por defecto de T5 (64).
- Entrada: `<bos>` + tokens v3 de las notas + `<eos>`. Target (labels): tokens
  TAB/TIME_SHIFT + `<eos>` (sin `<bos>`: T5 ya antepone su token de inicio del
  decoder, que aquí es `<pad>`).
- Predicción: por trozos según `data/chunking.chunk_plan` (paper: 20 notas con
  la última del trozo anterior como contexto en encoder y decoder; configurable:
  más contexto y anticipación), post-procesado por trozo (`postprocess.py`). Se
  hace por lotes: el trozo k de todas las pistas se genera a la vez (todos los
  trozos k>0 llevan un prefijo de igual longitud).
- Decodificación voraz (greedy) por defecto; `num_beams` opcional.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration

from ..data.chunking import PAPER_CONTEXT, PAPER_INFERENCE_CHUNK, chunk_plan
from ..fretboard import Fretboard
from ..metrics import difficulty_score, pitch_accuracy, tab_accuracy
from ..postprocess import match_window, postprocess, raw_alignment
from .constrained import make_prefix_fn
from ..tokenizer_v3 import EOS, V3Tokenizer

PAPER_CONFIG = dict(d_model=128, d_ff=1024, num_layers=3, num_decoder_layers=3, num_heads=4)


def build_model(tok: V3Tokenizer, **overrides) -> T5ForConditionalGeneration:
    cfg = T5Config(
        vocab_size=tok.vocab_size,
        decoder_start_token_id=tok.pad_id,
        pad_token_id=tok.pad_id,
        eos_token_id=tok.vocab[EOS],
        **{**PAPER_CONFIG, **overrides},
    )
    return T5ForConditionalGeneration(cfg)


def encode_pair(tok: V3Tokenizer, song: dict) -> Tuple[List[int], List[int]]:
    """(ids de entrada, labels) de una ventana."""
    src = tok.to_ids(tok.encode_input(song["notes"]))
    labels = tok.to_ids(tok.encode_target(song["tabs"]))[1:]  # sin <bos>
    return src, labels


class WindowDataset(Dataset):
    def __init__(self, windows: Sequence[dict], tok: V3Tokenizer):
        self.items = [encode_pair(tok, w) for w in windows]
        self.pad = tok.pad_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def collate(self, batch):
        smax = max(len(s) for s, _ in batch)
        tmax = max(len(t) for _, t in batch)
        src = torch.full((len(batch), smax), self.pad, dtype=torch.long)
        mask = torch.zeros((len(batch), smax), dtype=torch.long)
        labels = torch.full((len(batch), tmax), -100, dtype=torch.long)  # -100 = ignorado
        for i, (s, t) in enumerate(batch):
            src[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
            labels[i, :len(t)] = torch.tensor(t)
        return src, mask, labels


# --------------------------------------------------------------------------- #
# Predicción por trozos, por lotes, con post-procesado por etapas
# --------------------------------------------------------------------------- #
def _cut_eos(ids: List[int], eos: int) -> List[int]:
    return ids[: ids.index(eos)] if eos in ids else ids


@torch.no_grad()
def predict_tracks(
    model, tok: V3Tokenizer, tracks: Sequence[Sequence[Tuple[int, int]]],
    fb: Fretboard = None, chunk: int = PAPER_INFERENCE_CHUNK, ctx: int = PAPER_CONTEXT,
    commit: Optional[int] = None, num_beams: int = 1, batch_size: int = 128,
    constrained: bool = False,
) -> List[Dict[str, list]]:
    """Predice cada pista (lista de (pitch, dur)) por trozos según `chunk_plan`
    (`constrained=True`: decodificación restringida por pitch, ver constrained.py)
    (por defecto, el protocolo del paper: 20 notas + 1 de contexto).

    Devuelve, por pista, las tres etapas del post-procesado (una entrada por
    nota): "raw" (sin post-procesado), "window" (ventana ±5) y "final"
    (+ fallback). El post-procesado se aplica a todas las notas nuevas del trozo
    y se conservan las que se dan por buenas; la etapa final alimenta el
    contexto de los trozos siguientes.
    """
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
            srcs, decs, targets, keep = [], [], [], []
            for i in ids:
                cs, st, end, ce = plans[i][k]
                prefix = [(out[i]["final"][q], tracks[i][q][1]) for q in range(cs, st)]
                srcs.append(tok.to_ids(tok.encode_input(tracks[i][cs:end])))
                decs.append([tok.pad_id] + tok.to_ids(tok.encode_target(prefix), add_special=False))
                targets.append(tracks[i][st:end])
                keep.append(ce - st)
            smax = max(len(s) for s in srcs)
            src = torch.full((len(ids), smax), tok.pad_id, dtype=torch.long)
            mask = torch.zeros_like(src)
            for j, s in enumerate(srcs):
                src[j, :len(s)] = torch.tensor(s)
                mask[j, :len(s)] = 1
            dec = torch.tensor(decs)  # misma longitud en todo el lote (mismo k)
            extra = {}
            if constrained:
                extra["prefix_allowed_tokens_fn"] = make_prefix_fn(tok, fb, targets, dec.shape[1])
            gen = model.generate(
                input_ids=src.to(device), attention_mask=mask.to(device),
                decoder_input_ids=dec.to(device),
                max_new_tokens=2 * chunk + 4, do_sample=False, num_beams=num_beams, **extra,
            )
            for j, i in enumerate(ids):
                new = _cut_eos(gen[j, dec.shape[1]:].tolist(), eos)
                pred = tok.decode_target(tok.to_tokens(new))
                pitches = [p for p, _ in targets[j]]
                n = keep[j]
                out[i]["raw"] += raw_alignment(pred, len(pitches))[:n]
                out[i]["window"] += match_window(pred, pitches, fb)[:n]
                out[i]["final"] += postprocess(pred, pitches, fb)[:n]
    return out


def load_model(path: str) -> T5ForConditionalGeneration:
    return T5ForConditionalGeneration.from_pretrained(path)


def evaluate_tracks(model, tok: V3Tokenizer, songs: Sequence[dict], predict=None,
                    **kw) -> Dict[str, float]:
    """Métricas del paper por etapa de post-procesado, ponderadas por nota.
    `predict` permite reutilizarla con otros modelos (por defecto, el T5)."""
    fb = Fretboard()
    predict = predict or predict_tracks
    preds = predict(model, tok, [s["notes"] for s in songs], fb=fb, **kw)
    n = sum(len(s["notes"]) for s in songs)
    res = {"n_tracks": len(songs), "n_notes": n}
    for stage in ("raw", "window", "final"):
        res[f"tab_acc_{stage}"] = sum(
            tab_accuracy(p[stage], [t for t, _ in s["tabs"]]) * len(s["notes"])
            for p, s in zip(preds, songs)) / n
        res[f"pitch_acc_{stage}"] = sum(
            pitch_accuracy(p[stage], [q for q, _ in s["notes"]], fb) * len(s["notes"])
            for p, s in zip(preds, songs)) / n
    res["difficulty_pred"] = sum(
        difficulty_score(p["final"]) * len(s["notes"]) for p, s in zip(preds, songs)) / n
    res["difficulty_gt"] = sum(
        difficulty_score([t for t, _ in s["tabs"]]) * len(s["notes"]) for s in songs) / n
    return res
