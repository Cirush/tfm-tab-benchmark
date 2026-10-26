"""Variante propia de A* con función de coste diseñada y calibrada en este trabajo.

No es el A* del benchmark: el benchmark usa la función de coste del código original
de A-star-Guitar (`astar_burlet.py`). Esta variante se conserva porque su resultado
se cita como línea de trabajo futuro, y `AStarFretting` porque lo usan los datos
sintéticos de prueba.

Asignación por búsqueda de camino óptimo (inspirada en A-star-Guitar, Burlet & Fujinaga 2013).

Modela la secuencia de notas como un grafo por capas: cada capa contiene los
candidatos (cuerda, traste) de una nota; las aristas conectan candidatos de
notas consecutivas con un coste biomecánico. El camino de coste mínimo da la
digitación globalmente óptima.

Implementación: programación dinámica (Viterbi). Para grafos por capas es
equivalente a A*/Dijkstra y encuentra el mismo óptimo global de forma exacta,
con complejidad O(n · k²) siendo k ≤ 6 candidatos por nota.

Es una reimplementación propia a partir de la descripción del método, no el
código original. `AStarFretting` trata las notas de una en una. `AStarChords`
(al final del módulo) trata los acordes: cada capa son las digitaciones de las
notas simultáneas, como en A-star-Guitar.

Función de coste de `AStarFretting`:
  - w_move  · |Δfret| entre notas pulsadas consecutivas (movimiento de mano)
  - w_high  · exceso sobre el traste 12 (zona alta, más difícil)
  - w_open  · bonus (coste negativo) por cuerda al aire
  - w_string· |Δstring| (saltos grandes de cuerda penalizan ligeramente)
"""

from dataclasses import dataclass
from typing import List, Sequence
from ..fretboard import Fretboard, Position


@dataclass
class AStarWeights:
    w_move: float = 1.0
    w_high: float = 0.3
    w_open: float = -0.5
    w_string: float = 0.15


class AStarFretting:
    name = "astar"

    def __init__(self, fretboard: Fretboard | None = None,
                 weights: AStarWeights | None = None):
        self.fb = fretboard or Fretboard()
        self.w = weights or AStarWeights()

    # ---- costes ----
    def _node_cost(self, pos: Position) -> float:
        c = 0.0
        if pos.fret == 0:
            c += self.w.w_open
        elif pos.fret > 12:
            c += self.w.w_high * (pos.fret - 12)
        return c

    def _edge_cost(self, a: Position, b: Position) -> float:
        c = 0.0
        if a.fret > 0 and b.fret > 0:  # cuerdas al aire no mueven la mano
            c += self.w.w_move * abs(b.fret - a.fret)
        c += self.w.w_string * abs(b.string - a.string)
        return c

    # ---- inferencia ----
    def predict(self, pitches: Sequence[int]) -> List[Position]:
        if not pitches:
            return []
        layers: List[List[Position]] = []
        for p in pitches:
            cands = self.fb.candidates(p)
            if not cands:
                raise ValueError(f"Pitch {p} fuera del rango del instrumento")
            layers.append(cands)

        # DP hacia delante
        n = len(layers)
        cost = [[0.0] * len(layer) for layer in layers]
        back = [[-1] * len(layer) for layer in layers]

        for j, pos in enumerate(layers[0]):
            cost[0][j] = self._node_cost(pos)

        for i in range(1, n):
            for j, cur in enumerate(layers[i]):
                best_c, best_k = float("inf"), -1
                for k, prev in enumerate(layers[i - 1]):
                    c = cost[i - 1][k] + self._edge_cost(prev, cur)
                    if c < best_c:
                        best_c, best_k = c, k
                cost[i][j] = best_c + self._node_cost(cur)
                back[i][j] = best_k

        # reconstrucción del camino óptimo
        j = min(range(len(layers[-1])), key=lambda idx: cost[-1][idx])
        path = [layers[-1][j]]
        for i in range(n - 1, 0, -1):
            j = back[i][j]
            path.append(layers[i - 1][j])
        path.reverse()
        return path


# --------------------------------------------------------------------------- #
# Versión con acordes (polifónica), como A-star-Guitar
# --------------------------------------------------------------------------- #
import itertools

import numpy as np


@dataclass
class ChordWeights:
    """Costes de A-star-Guitar según los describe el Fretting-Transformer (§2):
    "the difficulty in moving between frets, the finger span required for
    chords, and penalties for positions beyond the seventh fret".

    Pesos calibrados con una búsqueda en rejilla sobre la validación del conjunto
    mezclado (nunca sobre test). Son los usados en los resultados finales; se
    pueden cambiar con `scripts/evaluate_benchmark.py --astar-weights`.
    """
    w_move: float = 0.5       # · |Δ posición de la mano| entre acordes consecutivos
    w_span: float = 1.0       # · apertura del acorde (traste máx − mín pisado)
    w_stretch: float = 1.0    # · (apertura − 3)² si supera 3 trastes (incómodo)
    w_high: float = 0.3       # · trastes por encima del 7
    w_open: float = 0.0       # · nº de cuerdas al aire


class AStarChords:
    """Camino óptimo sobre ACORDES: cada capa son las digitaciones posibles de
    las notas que suenan a la vez (una nota por cuerda). Viterbi exacto sobre
    las `top_k` digitaciones más cómodas de cada acorde (poda)."""
    name = "astar_chords"

    def __init__(self, fretboard: Fretboard | None = None, weights: ChordWeights | None = None,
                 top_k: int = 30, onset_tol: int = 120):
        self.fb = fretboard or Fretboard()
        self.w = weights or ChordWeights()
        self.top_k = top_k
        self.onset_tol = onset_tol  # ticks: notas que empiezan a < 1 fusa = mismo acorde (rasgueos)

    def groups(self, starts: Sequence[int]) -> List[List[int]]:
        """Índices de notas agrupados en acordes por instante de inicio."""
        out, cur, t0 = [], [], None
        for i, t in enumerate(starts):
            if cur and t - t0 > self.onset_tol:
                out.append(cur)
                cur = []
            if not cur:
                t0 = t
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def _fingerings(self, pitches: Sequence[int]):
        """Digitaciones de un acorde (cuerdas distintas) ordenadas por coste propio."""
        cands = [self.fb.candidates(p) for p in pitches]
        if any(not c for c in cands):
            raise ValueError("pitch fuera del rango del instrumento")
        if np.prod([len(c) for c in cands]) > 50_000:
            # acordes muy densos: solo las 3 posiciones más bajas de cada nota
            cands = [sorted(c, key=lambda p: p.fret)[:3] for c in cands]
        combos = [c for c in itertools.product(*cands)
                  if len({pos.string for pos in c}) == len(c)]
        if not combos:  # imposible con cuerdas distintas (p. ej. >6 notas): se relaja
            combos = list(itertools.product(*cands))
        scored = sorted(((self._chord_cost(c), c) for c in combos), key=lambda x: x[0])[: self.top_k]
        return [c for _, c in scored], np.array([s for s, _ in scored])

    def _chord_cost(self, chord) -> float:
        frets = [p.fret for p in chord if p.fret > 0]
        span = max(frets) - min(frets) if frets else 0
        w = self.w
        return (w.w_span * span + w.w_stretch * max(0, span - 3) ** 2
                + w.w_high * sum(max(0, p.fret - 7) for p in chord)
                + w.w_open * sum(p.fret == 0 for p in chord))

    @staticmethod
    def _hand(chord) -> float:
        """Posición de la mano: traste pisado más bajo (NaN si todo al aire)."""
        frets = [p.fret for p in chord if p.fret > 0]
        return min(frets) if frets else np.nan

    def predict(self, pitches: Sequence[int], starts: Sequence[int] | None = None) -> List[Position]:
        if not pitches:
            return []
        starts = list(starts) if starts is not None else list(range(0, len(pitches) * 10**6, 10**6))
        groups = self.groups(starts)
        layers, node, hands = [], [], []
        for g in groups:
            fings, cost = self._fingerings([pitches[i] for i in g])
            layers.append(fings)
            node.append(cost)
            hands.append(np.array([self._hand(f) for f in fings]))
        # Viterbi: coste de transición = w_move · |Δ mano| (0 si alguno va al aire)
        acc, back = node[0].copy(), []
        for t in range(1, len(layers)):
            d = np.abs(hands[t - 1][:, None] - hands[t][None, :])
            edge = self.w.w_move * np.nan_to_num(d, nan=0.0)
            total = acc[:, None] + edge
            back.append(total.argmin(axis=0))
            acc = total.min(axis=0) + node[t]
        j = int(acc.argmin())
        chosen = [j]
        for t in range(len(layers) - 1, 0, -1):
            j = int(back[t - 1][j])
            chosen.append(j)
        chosen.reverse()
        out: List[Position] = [None] * len(pitches)
        for g, fings, j in zip(groups, layers, chosen):
            for i, pos in zip(g, fings[j]):
                out[i] = pos
        return out
