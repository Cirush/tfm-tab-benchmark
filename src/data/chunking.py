"""Troceo de pistas en secuencias para los modelos seq2seq (Fretting-Transformer).

Paper (arXiv:2506.14223):
- Entrenamiento: *"Both models were trained on input sequences of 512 tokens
  length."* No especifica cómo se parte una pista.
- Inferencia: *"chunks of 20 notes are processed. The tokens from the last note
  of the previous chunk are placed at the beginning of the following sequence in
  both the encoder and decoder to preserve context between the chunks."*

Decisiones:
- Entrenamiento: ventanas consecutivas sin solape de hasta `max_notes_for(512)`
  notas completas. Entrada y target se cortan por la misma frontera de notas, así
  nunca se desalinean (truncar cada secuencia por nº de tokens sí los desalinearía).
- Entrenamiento alternativo: `random_windows` (longitud aleatoria) para que un
  único modelo sirva con distintos protocolos de inferencia.
- Inferencia: `chunk_plan` define los trozos; `chunked_predict` lo aplica con
  cualquier predictor (testeable sin modelo). Parámetros:
    chunk  = notas nuevas que ve el modelo por trozo,
    ctx    = notas ya decididas que se repiten como contexto (encoder y decoder),
    commit = cuántas de las notas nuevas se dan por buenas (≤ chunk); si es menor
             que chunk, el resto solo sirve de anticipación (lookahead).
  Protocolo del paper: chunk=20, ctx=1, commit=20.
- El troceo se aplica después del split por canción (ver `dadagp_gp.split_paper`).
"""

import random
from typing import Callable, List, Optional, Sequence, Tuple

from ..fretboard import Position

# Tokens por nota en el encoding v3: NOTE_ON, TIME_SHIFT, NOTE_OFF (entrada) y
# TAB, TIME_SHIFT (target). Más <bos> y <eos> por secuencia.
INPUT_TOKENS_PER_NOTE = 3
SPECIAL_TOKENS = 2

PAPER_MAX_TOKENS = 512
PAPER_INFERENCE_CHUNK = 20
PAPER_CONTEXT = 1

Note = Tuple[int, int]            # (pitch, dur)
Tab = Tuple[Position, int]        # (posición, dur)


def max_notes_for(max_tokens: int = PAPER_MAX_TOKENS) -> int:
    """Nº máximo de notas completas cuya entrada cabe en `max_tokens` (512 -> 170)."""
    return (max_tokens - SPECIAL_TOKENS) // INPUT_TOKENS_PER_NOTE


def training_windows(
    song: dict, max_notes: Optional[int] = None, min_notes: int = 1
) -> List[dict]:
    """Parte una pista en ventanas consecutivas sin solape de ≤ max_notes notas.

    Cada ventana conserva la correspondencia 1:1 nota↔TAB y añade a `meta` la
    posición de la ventana (`window`, `start`). Las ventanas finales con menos de
    `min_notes` notas se descartan.
    """
    max_notes = max_notes or max_notes_for()
    notes, tabs = song["notes"], song["tabs"]
    assert len(notes) == len(tabs), "pista desalineada antes de trocear"
    out = []
    for k, start in enumerate(range(0, len(notes), max_notes)):
        end = start + max_notes
        if len(notes[start:end]) < min_notes:
            continue
        out.append({
            "notes": list(notes[start:end]),
            "tabs": list(tabs[start:end]),
            "meta": {**song.get("meta", {}), "window": k, "start": start},
        })
    return out


def random_windows(
    song: dict, min_notes: int, max_notes: int, rng: random.Random
) -> List[dict]:
    """Parte una pista en ventanas consecutivas sin solape de longitud aleatoria
    uniforme en [min_notes, max_notes] (la última puede ser más corta)."""
    notes, tabs = song["notes"], song["tabs"]
    assert len(notes) == len(tabs), "pista desalineada antes de trocear"
    assert max_notes <= max_notes_for(), "la ventana no cabe en 512 tokens"
    out, start, k = [], 0, 0
    while start < len(notes):
        end = start + rng.randint(min_notes, max_notes)
        out.append({
            "notes": list(notes[start:end]),
            "tabs": list(tabs[start:end]),
            "meta": {**song.get("meta", {}), "window": k, "start": start},
        })
        start, k = end, k + 1
    return out


def chunk_plan(
    n: int, chunk: int = PAPER_INFERENCE_CHUNK, ctx: int = PAPER_CONTEXT,
    commit: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """Trozos de inferencia para una pista de n notas: [(ctx_start, start, end,
    commit_end)]. El modelo ve notes[ctx_start:end]; notes[ctx_start:start] es
    contexto ya decidido (prefijo del decoder); se dan por buenas
    notes[start:commit_end]. El último trozo da por buenas todas sus notas.

    Con commit ≥ ctx, todos los trozos salvo el primero tienen exactamente `ctx`
    notas de contexto (permite generar por lotes con prefijos de igual longitud).
    """
    commit = commit or chunk
    assert 0 < commit <= chunk and commit >= ctx >= 0
    plan, p = [], 0
    while p < n:
        end = min(n, p + chunk)
        commit_end = end if end == n else p + commit
        plan.append((max(0, p - ctx), p, end, commit_end))
        p = commit_end
    return plan


# Predictor: (notas de la ventana, prefijo de TABs ya decididos para las
# primeras notas) -> posiciones para las notas restantes (tras el prefijo).
Predictor = Callable[[Sequence[Note], Sequence[Tab]], List[Position]]


def chunked_predict(
    notes: Sequence[Note], predict: Predictor, chunk: int = PAPER_INFERENCE_CHUNK,
    ctx: int = PAPER_CONTEXT, commit: Optional[int] = None,
) -> List[Position]:
    """Inferencia por trozos según `chunk_plan` (por defecto, como el paper).

    Cada trozo recibe como entrada [notas de contexto] + sus notas, y como
    prefijo del decoder los TABs ya decididos para el contexto. Se conservan
    solo las posiciones de las notas que se dan por buenas, de modo que el
    resultado tiene una posición por nota de entrada... siempre que el predictor
    devuelva una por nota (lo garantiza el post-procesado).
    """
    out: List[Position] = []
    for cs, st, end, ce in chunk_plan(len(notes), chunk, ctx, commit):
        prefix = [(out[i], notes[i][1]) for i in range(cs, st)]
        pred = predict(list(notes[cs:end]), prefix)
        out.extend(pred[: ce - st])
    return out
