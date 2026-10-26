"""Decodificación restringida por pitch (Hsu et al. 2026, arXiv 2607.26440, §III-4).

En cada paso en que el modelo genera un TAB, solo se permiten las posiciones que
producen la nota de entrada: T(p) = {(s, f) | afinación[s] + f = p}. Además se
fuerza la estructura de la entrada: tras cada TAB va el TIME_SHIFT de esa nota y,
tras la última nota, <eos>. El modelo no puede equivocarse de nota ni
desalinearse; solo elige entre posiciones válidas (la ambigüedad real).

A diferencia del post-procesado del Fretting-Transformer (que corrige después de
generar), aquí el error no puede ocurrir y no se arrastra a los pasos siguientes.
No requiere reentrenar: es solo otra forma de generar.
"""

from typing import Callable, Dict, List, Sequence, Tuple

from ..fretboard import Fretboard
from ..tokenizer_v3 import EOS, V3Tokenizer


def valid_tab_ids(tok: V3Tokenizer, fb: Fretboard, pitch: int) -> List[int]:
    """Ids de los tokens TAB que producen `pitch` en el mástil `fb`."""
    return [tok.vocab[f"TAB<{p.string},{p.fret}>"] for p in fb.candidates(pitch)
            if f"TAB<{p.string},{p.fret}>" in tok.vocab]


def allowed_tokens(tok: V3Tokenizer, fb: Fretboard, notes: Sequence[Tuple[int, int]],
                   step: int, cache: Dict[int, List[int]]) -> List[int]:
    """Tokens permitidos en el paso `step` (contado desde el primer token a
    generar) para la secuencia de notas `notes`: TAB válido, TIME_SHIFT de la
    nota o <eos> al terminar."""
    eos = tok.vocab[EOS]
    if step >= 2 * len(notes):
        return [eos, tok.pad_id]
    pitch, dur = notes[step // 2]
    if step % 2 == 0:
        if pitch not in cache:
            cache[pitch] = valid_tab_ids(tok, fb, pitch)
        return cache[pitch]
    return [tok.vocab[f"TIME_SHIFT<{tok.quantize(dur)}>"]]


def make_prefix_fn(tok: V3Tokenizer, fb: Fretboard,
                   notes_per_item: Sequence[Sequence[Tuple[int, int]]],
                   start_len: int) -> Callable:
    """`prefix_allowed_tokens_fn` para `model.generate`: `start_len` es la
    longitud de la secuencia antes del primer token generado (decoder del T5 con
    su prefijo de contexto, o prompt rellenado del GPT2)."""
    cache: Dict[int, List[int]] = {}

    def fn(batch_id: int, input_ids) -> List[int]:
        return allowed_tokens(tok, fb, notes_per_item[batch_id],
                              input_ids.shape[-1] - start_len, cache)
    return fn
