"""Baseline: traste mínimo.

Para cada nota, elige la posición candidata con menor traste (prefiriendo
cuerdas al aire). En caso de empate, prefiere la cuerda más grave (número
mayor), heurística común porque deja las agudas libres para la melodía.

Es la referencia mínima de la comparativa: sin contexto, decide nota a nota.
Equivale al baseline del Fretting-Transformer.
"""

from typing import List, Sequence
from ..fretboard import Fretboard, Position


class MinFretBaseline:
    name = "baseline_minfret"

    def __init__(self, fretboard: Fretboard | None = None):
        self.fb = fretboard or Fretboard()

    def predict(self, pitches: Sequence[int]) -> List[Position]:
        """Asigna posición a cada pitch de forma independiente (sin contexto)."""
        out: List[Position] = []
        for p in pitches:
            cands = self.fb.candidates(p)
            if not cands:
                raise ValueError(f"Pitch {p} fuera del rango del instrumento")
            # menor traste; empate -> cuerda más grave (string mayor)
            best = min(cands, key=lambda c: (c.fret, -c.string))
            out.append(best)
        return out
