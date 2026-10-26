"""Métricas unificadas del benchmark.

- tab_accuracy: % de notas con (cuerda, traste) exactamente iguales al ground truth.
  Es la métrica principal del Fretting-Transformer.
- string_accuracy / fret_accuracy: desgloses útiles para el análisis.
- pitch_accuracy: % de notas cuyo TAB suena a la nota pedida (paper).
- difficulty_score: dificultad de digitación del paper (arXiv:2506.14223, ec. 1-6).

Notas sobre el difficulty score:
- Se implementan las ecuaciones tal cual, con dos matices documentados:
  1. El paper define Δcuerda = q − p con signo; se usa |Δcuerda| porque con
     signo un salto de la 6ª a la 1ª (−5 ≤ 1) costaría como no moverse. En el
     traste el signo sí es intencionado (subir 0,50 / bajar 0,75 por traste).
  2. El paper dice que el rango es 0-18,5 en guitarra de 24 trastes. Con sus
     ecuaciones, 18,5 es subir del traste 0 al 24 (12 + 6 + 0,5), pero bajar
     del 24 al 0 da 24,5 y el mínimo es 0,25 → el rango publicado es una errata.
- Las cuerdas al aire no tienen trato especial (el paper no lo indica).
"""

from typing import Optional, Sequence
from .fretboard import Fretboard, Position


def tab_accuracy(pred: Sequence[Optional[Position]], gt: Sequence[Position]) -> float:
    """Fracción de notas con cuerda Y traste correctos. Requiere igual longitud.
    Una predicción None (nota sin TAB, p. ej. salida sin post-procesar) es fallo."""
    if len(pred) != len(gt):
        raise ValueError(f"Longitudes distintas: pred={len(pred)}, gt={len(gt)}")
    if not gt:
        return 0.0
    hits = sum(1 for p, g in zip(pred, gt) if p is not None and p == g)
    return hits / len(gt)


def pitch_accuracy(
    pred: Sequence[Optional[Position]], pitches: Sequence[int], fb: Fretboard
) -> float:
    """Fracción de notas cuyo TAB predicho suena al pitch de entrada (paper:
    admite posiciones alternativas mientras la nota sea la correcta)."""
    if len(pred) != len(pitches):
        raise ValueError(f"Longitudes distintas: pred={len(pred)}, notas={len(pitches)}")
    if not pitches:
        return 0.0
    hits = sum(
        1 for p, pitch in zip(pred, pitches)
        if p is not None and 1 <= p.string <= len(fb.tuning) and fb.pitch_of(p) == pitch
    )
    return hits / len(pitches)


def string_accuracy(pred: Sequence[Optional[Position]], gt: Sequence[Position]) -> float:
    """Fracción de notas con la cuerda correcta (independiente del traste)."""
    if len(pred) != len(gt):
        raise ValueError("Longitudes distintas")
    if not gt:
        return 0.0
    return sum(1 for p, g in zip(pred, gt) if p is not None and p.string == g.string) / len(gt)


# Pesos del difficulty score del paper (Radicioni & Lombardo, modificado).
FRET_UP, FRET_DOWN = 0.50, 0.75
LOCALITY_ALPHA = 0.25
STRING_NEAR, STRING_FAR = 0.25, 0.50


def difficulty(p: Position, q: Position) -> float:
    """Dificultad de pasar de la posición p a la q (ecuaciones 1-6 del paper).

    difficulty = along + across
      along  = fret_stretch + locality
        fret_stretch = 0.50·|Δfret| si Δfret > 0 (subir), 0.75·|Δfret| si ≤ 0
        locality     = 0.25·(fret_p + fret_q)       (penaliza trastes altos)
      across = vertical_stretch = 0.25 si |Δcuerda| ≤ 1, 0.50 si > 1
    """
    dfret = q.fret - p.fret
    fret_stretch = (FRET_UP if dfret > 0 else FRET_DOWN) * abs(dfret)
    locality = LOCALITY_ALPHA * (p.fret + q.fret)
    vertical = STRING_NEAR if abs(q.string - p.string) <= 1 else STRING_FAR
    return fret_stretch + locality + vertical


def difficulty_score(seq: Sequence[Position]) -> float:
    """Difficulty score de una tablatura: media de `difficulty` sobre las
    transiciones entre notas consecutivas (paper: "mean difficulty over all
    positions"). Menor = más fácil. 0.0 si hay menos de 2 notas."""
    if len(seq) < 2:
        return 0.0
    return sum(difficulty(p, q) for p, q in zip(seq, seq[1:])) / (len(seq) - 1)


def evaluate(pred: Sequence[Position], gt: Sequence[Position]) -> dict:
    """Evalúa una secuencia predicha frente al ground truth. Devuelve dict de métricas."""
    return {
        "tab_accuracy": tab_accuracy(pred, gt),
        "string_accuracy": string_accuracy(pred, gt),
        "difficulty_pred": difficulty_score(pred),
        "difficulty_gt": difficulty_score(gt),
        "n_notes": len(gt),
    }
