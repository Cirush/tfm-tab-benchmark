"""Post-procesado de la salida de los modelos generativos (Fretting-Transformer §3.5).

Problema: un modelo seq2seq genera libremente, así que su salida puede tener
más o menos TABs que notas de entrada, o TABs que no producen la nota pedida.

Paper (arXiv:2506.14223): *"Our post-processing algorithm refines the model's
output by comparing the estimated note sequence to the corresponding input note
sequence. It attempts to match each input note to its closest counterpart in the
estimated sequence within a configurable window of ±5 notes. The algorithm
evaluates pitch values and selects the best match. If no direct match is found,
the first viable string-fret combination generated for the guitar configuration
used is applied."*

Su Tabla 2 reporta tres variantes, que aquí son tres etapas medibles por separado:
  sin post-procesado          -> `raw_alignment`   (97,23% pitch / 68,56% tab)
  + overlap correction        -> `match_window`    (99,92% / 72,15%)
  + neighbor search           -> `postprocess`     (100%   / 72,19%)

Interpretación (el paper no da más detalle; decisiones documentadas):
- "Nota estimada" = pitch que suena con el TAB predicho según la afinación.
- La ventana ±5 se indexa por posición de nota: para la nota de entrada i se
  miran las predicciones j ∈ [i-5, i+5] (lectura literal del paper).
- "Mejor match" = el TAB con el mismo pitch más cercano a i; en empate, el
  anterior (j < i).
- Alternativa evaluada y DESCARTADA: centrar la ventana en i + desfase (seguir
  el desplazamiento que provocan notas saltadas/insertadas). Simulando errores
  sobre ~310k notas reales de validación (4 escenarios, 2-25% de error;
  `scripts/simulate_postprocess.py`), fue siempre peor (-0,3 a -0,5 puntos de
  tab accuracy): una coincidencia casual fija un
  desfase erróneo que arrastra a las notas siguientes, mientras que en música
  real la nota vecina con el mismo pitch suele tener la misma posición.
  Limitación conocida de la versión literal: tras un desplazamiento, una vecina
  con el mismo pitch y distinta posición puede "robar" el match
  (ver test_limitacion_conocida_vecina_mismo_pitch).
- Un TAB puede reutilizarse para varias notas (notas repetidas son frecuentes;
  forzar unicidad propagaría errores).
- "Primera combinación viable" = primer candidato de `Fretboard.candidates`
  (ordenados de la cuerda 1, aguda, a la 6).
"""

from typing import List, Optional, Sequence

from .fretboard import Fretboard, Position

PAPER_WINDOW = 5


def _valid(pos: Position, fb: Fretboard) -> bool:
    return 1 <= pos.string <= len(fb.tuning) and 0 <= pos.fret <= fb.num_frets


def raw_alignment(pred: Sequence[Position], n_notes: int) -> List[Optional[Position]]:
    """Sin post-procesado: la predicción j se asigna a la nota j tal cual.

    Recorta si sobran TABs y rellena con None (= fallo) si faltan.
    """
    out: List[Optional[Position]] = list(pred[:n_notes])
    return out + [None] * (n_notes - len(out))


def match_window(
    pred: Sequence[Position], pitches: Sequence[int], fb: Fretboard,
    window: int = PAPER_WINDOW,
) -> List[Optional[Position]]:
    """Overlap correction: a cada nota le asigna el TAB predicho más cercano
    (dentro de ±window) que produce su pitch. None si no hay ninguno."""
    offsets = sorted(range(-window, window + 1), key=lambda d: (abs(d), d))
    out: List[Optional[Position]] = []
    for i, pitch in enumerate(pitches):
        match = None
        for d in offsets:
            j = i + d
            if 0 <= j < len(pred) and _valid(pred[j], fb) and fb.pitch_of(pred[j]) == pitch:
                match = pred[j]
                break
        out.append(match)
    return out


def fill_fallback(
    aligned: Sequence[Optional[Position]], pitches: Sequence[int], fb: Fretboard
) -> List[Position]:
    """Neighbor search / fallback: las notas sin match reciben la primera
    posición viable para su pitch. Falla si un pitch no cabe en el mástil."""
    out = []
    for pos, pitch in zip(aligned, pitches):
        if pos is None:
            cands = fb.candidates(pitch)
            if not cands:
                raise ValueError(f"pitch {pitch} fuera del rango del mástil")
            pos = cands[0]
        out.append(pos)
    return out


def postprocess(
    pred: Sequence[Position], pitches: Sequence[int], fb: Fretboard,
    window: int = PAPER_WINDOW,
) -> List[Position]:
    """Post-procesado completo: exactamente 1 posición por nota y todas suenan
    al pitch pedido (100% pitch accuracy por construcción)."""
    return fill_fallback(match_window(pred, pitches, fb, window), pitches, fb)
