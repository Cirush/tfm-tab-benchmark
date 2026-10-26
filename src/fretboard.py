"""Modelo del mástil de guitarra.

Representa la afinación y calcula las posiciones (cuerda, traste) candidatas
para un pitch MIDI dado. Es la base de todos los modelos del benchmark.

Convención de cuerdas: 1 = cuerda más aguda (Mi4), 6 = más grave (Mi2),
igual que en notación de tablatura estándar.
"""

from dataclasses import dataclass
from typing import List, Tuple

# Afinación estándar: pitch MIDI de cada cuerda al aire, índice 0 = cuerda 1 (aguda)
# E4=64, B3=59, G3=55, D3=50, A2=45, E2=40
STANDARD_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)

# Drop D: la 6ª cuerda baja a D2=38 (común en DadaGP, dominado por rock/metal)
DROP_D_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 38)


@dataclass(frozen=True)
class Position:
    """Una posición en el mástil: cuerda (1-6) y traste (0 = al aire)."""
    string: int
    fret: int

    def __repr__(self) -> str:
        return f"({self.string},{self.fret})"


class Fretboard:
    """Mástil con una afinación y nº de trastes; genera candidatos por pitch."""

    def __init__(self, tuning: Tuple[int, ...] = STANDARD_TUNING, num_frets: int = 24):
        self.tuning = tuning
        self.num_frets = num_frets

    def candidates(self, pitch: int) -> List[Position]:
        """Todas las posiciones (cuerda, traste) que producen el pitch dado.

        Devuelve lista vacía si el pitch está fuera del rango del instrumento.
        Ordenadas por número de cuerda (1º la más aguda).
        """
        result = []
        for i, open_pitch in enumerate(self.tuning):
            fret = pitch - open_pitch
            if 0 <= fret <= self.num_frets:
                result.append(Position(string=i + 1, fret=fret))
        return result

    def pitch_of(self, pos: Position) -> int:
        """Pitch MIDI que produce una posición dada."""
        return self.tuning[pos.string - 1] + pos.fret

    def in_range(self, pitch: int) -> bool:
        return len(self.candidates(pitch)) > 0
