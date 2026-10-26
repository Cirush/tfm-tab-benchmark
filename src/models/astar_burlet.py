"""A-star-Guitar con la función de coste del código original de Burlet y Fujinaga (2013).

Traducción a este proyecto de la función de coste publicada por los autores en
https://github.com/gburlet/astar-guitar (licencia MIT; `astar.py`,
`biomechanical_burlet`, y `guitar/guitarevent.py`, métodos `distance`), en su
versión final. No se copia su código: se reproduce su comportamiento sobre la
representación de este proyecto. Los pesos son los del original, sin calibrar.

Cada evento (nota suelta o acorde) es una capa del grafo. El coste de pasar del
candidato n1 al candidato n2 es

    2 · distancia(n1, n2) + 1 · [traste máximo de n2 > 7]
      + 2 · apertura(n2) + 1 · cuerdas saltadas dentro de n2

  - distancia (0 si n1 va todo al aire):
      nota → nota:       |f1 − f2|, o 0 si alguna está al aire
      acorde → acorde:   |máx1 − máx2|, o 0 si algún máximo es 0
      nota ↔ acorde:     si la nota cae fuera del rango [mín, máx] del acorde,
                         distancia al borde más cercano; si cae dentro, distancia
                         al centro (mín+máx)//2
  - apertura: máx − mín de todos los trastes del acorde (cuerdas al aire incluidas)
  - cuerdas saltadas: cuerdas sin tocar entre la más aguda y la más grave del acorde

Las digitaciones de acorde exigen cuerdas distintas y como mucho 7 trastes de
apertura entre las notas pisadas. Los mínimos, máximos y el centro incluyen las
cuerdas al aire, como en el original (división entera de Python 2).

El original busca con `networkx.astar_path` sin heurística, que equivale a
Dijkstra. Como el coste solo depende de dos eventos consecutivos, aquí se usa el
mismo Viterbi exacto que en `astar.py`: el óptimo es el mismo y solo puede
cambiar la elección entre caminos empatados.

Diferencias inevitables con el original:
  - Los acordes se agrupan por instante de inicio (como `AStarChords`), no con
    las marcas de acorde de una partitura MusicXML/MEI.
  - El original descarta los eventos sin digitación válida (más de 6 notas o
    notas que solo caben en la misma cuerda) y los deja sin tablatura. Aquí
    también se quitan del grafo, pero sus notas reciben el traste más bajo
    posible, para que todos los métodos den una posición por nota.
"""

from functools import lru_cache
from typing import List, Sequence

import itertools

import numpy as np

from ..fretboard import Fretboard, Position
from .astar import AStarChords

W_DISTANCE = 2
W_FRET_PENALTY = 1
FRET_THRESHOLD = 7
W_CHORD_DISTANCE = 2
W_CHORD_STRING_DISTANCE = 1
MAX_CHORD_SPAN = 7


class AStarBurlet:
    name = "astar_burlet"

    def __init__(self, fretboard: Fretboard | None = None, onset_tol: int = 120):
        self.fb = fretboard or Fretboard()
        self.onset_tol = onset_tol
        self._candidates = lru_cache(maxsize=None)(self._candidates_uncached)

    def groups(self, starts: Sequence[int]) -> List[List[int]]:
        return AStarChords.groups(self, starts)

    # ---- candidatos de un evento ----
    def _candidates_uncached(self, pitches: tuple):
        """Digitaciones válidas de un evento y sus rasgos para el coste.

        Devuelve (digitaciones, rasgos) con rasgos = arrays de f (traste si es
        nota suelta), mín, máx, todo_al_aire y coste propio del candidato.
        """
        cands = [self.fb.candidates(p) for p in pitches]
        if any(not c for c in cands):
            raise ValueError("pitch fuera del rango del instrumento")
        if len(pitches) == 1:
            fings = [(c,) for c in cands[0]]
        elif len(pitches) > 6:
            fings = []
        else:
            fings = []
            for c in itertools.product(*cands):
                if len({p.string for p in c}) < len(c):
                    continue
                pressed = [p.fret for p in c if p.fret > 0]
                if pressed and max(pressed) - min(pressed) > MAX_CHORD_SPAN:
                    continue
                fings.append(c)
        if not fings:
            return [], None
        frets = [[p.fret for p in c] for c in fings]
        mn = np.array([min(f) for f in frets])
        mx = np.array([max(f) for f in frets])
        node = W_FRET_PENALTY * (mx > FRET_THRESHOLD)
        if len(pitches) > 1:
            span = mx - mn
            strings = [[p.string for p in c] for c in fings]
            skipped = np.array([max(s) - min(s) - (len(s) - 1) for s in strings])
            node = node + W_CHORD_DISTANCE * span + W_CHORD_STRING_DISTANCE * skipped
        feats = {"f": mx, "mn": mn, "mx": mx, "open": mx == 0, "node": node.astype(float)}
        return fings, feats

    # ---- distancia entre dos capas ----
    @staticmethod
    def _note_to_chord(f, mn, mx):
        """Distancia de Burlet entre una nota en el traste f y un acorde [mn, mx]."""
        inside = f - (mn + mx) // 2
        d = np.where(f <= mn, mn - f, np.where(f >= mx, f - mx, inside))
        return np.abs(d)

    def _distance(self, a, a_chord, b, b_chord):
        """Matriz (candidatos de a) × (candidatos de b) de la distancia del original."""
        if not a_chord and not b_chord:
            fa, fb_ = a["f"][:, None], b["f"][None, :]
            d = np.where((fa == 0) | (fb_ == 0), 0, np.abs(fa - fb_))
        elif a_chord and b_chord:
            ma, mb = a["mx"][:, None], b["mx"][None, :]
            d = np.where((ma == 0) | (mb == 0), 0, np.abs(ma - mb))
        elif not a_chord:
            d = self._note_to_chord(a["f"][:, None], b["mn"][None, :], b["mx"][None, :])
        else:
            d = self._note_to_chord(b["f"][None, :], a["mn"][:, None], a["mx"][:, None])
        return np.where(a["open"][:, None], 0, d)

    def predict(self, pitches: Sequence[int], starts: Sequence[int] | None = None) -> List[Position]:
        if not pitches:
            return []
        starts = list(starts) if starts is not None else list(range(0, len(pitches) * 10**6, 10**6))
        out: List[Position] = [None] * len(pitches)
        layers = []  # (índices de las notas, digitaciones, rasgos, es_acorde)
        for g in self.groups(starts):
            ps = [pitches[i] for i in g]
            fings, feats = self._candidates(tuple(ps))
            if not fings:  # el original deja el evento sin tablatura
                for i, p in zip(g, ps):
                    out[i] = min(self.fb.candidates(p), key=lambda pos: pos.fret)
                continue
            layers.append((g, fings, feats, len(g) > 1))
        if not layers:
            return out

        # Viterbi: desde el nodo inicial solo cuenta el coste propio del candidato
        acc, back = layers[0][2]["node"].copy(), []
        for (_, _, fa, ca), (_, _, fb_, cb) in zip(layers, layers[1:]):
            total = acc[:, None] + W_DISTANCE * self._distance(fa, ca, fb_, cb) + fb_["node"][None, :]
            back.append(total.argmin(axis=0))
            acc = total.min(axis=0)
        j = int(acc.argmin())
        chosen = [j]
        for t in range(len(layers) - 1, 0, -1):
            j = int(back[t - 1][j])
            chosen.append(j)
        chosen.reverse()
        for (g, fings, _, _), j in zip(layers, chosen):
            for i, pos in zip(g, fings[j]):
                out[i] = pos
        return out
