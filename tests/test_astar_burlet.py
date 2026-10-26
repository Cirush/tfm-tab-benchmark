"""Tests de A-star-Guitar con la función de coste original (src/models/astar_burlet.py)."""

import itertools

import numpy as np

from src.fretboard import Fretboard, Position
from src.models.astar_burlet import AStarBurlet


def coste_original(n1, n2):
    """Coste de arista escrito como en `biomechanical_burlet` del código original.
    n1, n2: tupla de Position (1 elemento = nota suelta); n1 = None es el inicio."""
    distancia = 0
    if n1 is not None and not all(p.fret == 0 for p in n1):
        f1 = [p.fret for p in n1]
        f2 = [p.fret for p in n2]
        if len(n1) == 1 and len(n2) == 1:
            distancia = 0 if f1[0] == 0 or f2[0] == 0 else abs(f1[0] - f2[0])
        elif len(n1) > 1 and len(n2) > 1:
            distancia = 0 if max(f1) == 0 or max(f2) == 0 else abs(max(f1) - max(f2))
        else:
            f, acorde = (f1[0], f2) if len(n1) == 1 else (f2[0], f1)
            lo, hi = min(acorde), max(acorde)
            if f <= lo:
                distancia = lo - f
            elif f >= hi:
                distancia = f - hi
            else:
                distancia = abs(f - (lo + hi) // 2)
    frets = [p.fret for p in n2]
    penal = 1 if max(frets) > 7 else 0
    apertura = saltadas = 0
    if len(n2) > 1:
        apertura = max(frets) - min(frets)
        s = sorted(p.string for p in n2)
        saltadas = s[-1] - s[0] - (len(s) - 1)
    return 2 * distancia + penal + 2 * apertura + saltadas


def test_distancia_nota_acorde_como_el_original():
    m = AStarBurlet()
    d = m._note_to_chord(np.array([0, 2, 5, 9]), np.array(3), np.array(7))
    # por debajo: 3 − f; por encima: f − 7; dentro: |f − (3+7)//2|
    assert d.tolist() == [3, 1, 0, 2]


def test_coste_propio_del_acorde():
    m = AStarBurlet()
    # E5 en Mi grave: cuerdas 6 (traste 0) y 5 (traste 2)
    fings, feats = m._candidates((40, 47))
    i = fings.index((Position(6, 0), Position(5, 2)))
    assert feats["node"][i] == 2 * 2  # apertura 2 (el aire cuenta como traste 0), sin saltos
    # notas en cuerdas 1 y 3: una cuerda saltada
    fings, feats = m._candidates((64, 55))
    i = fings.index((Position(1, 0), Position(3, 0)))
    assert feats["node"][i] == 1


def test_viterbi_igual_que_busqueda_exhaustiva_con_el_coste_original():
    fb = Fretboard()
    m = AStarBurlet(fb)
    pitches = [52, 55, 59, 60, 64, 67, 62, 57]
    starts = [0, 0, 0, 960, 1440, 1440, 1920, 2400]
    eventos = m.groups(starts)
    opciones = [m._candidates(tuple(pitches[i] for i in g))[0] for g in eventos]
    mejor = min(itertools.product(*opciones),
                key=lambda camino: sum(coste_original(a, b) for a, b in zip((None,) + camino, camino)))
    coste_mejor = sum(coste_original(a, b) for a, b in zip((None,) + mejor, mejor))
    pred = m.predict(pitches, starts)
    camino_pred = tuple(tuple(pred[i] for i in g) for g in eventos)
    assert sum(coste_original(a, b) for a, b in zip((None,) + camino_pred, camino_pred)) == coste_mejor
    assert [fb.pitch_of(p) for p in pred] == pitches


def test_acordes_validos_y_evento_sin_digitacion():
    fb = Fretboard()
    m = AStarBurlet(fb)
    # Do mayor abierto + acorde de 7 notas (no cabe en 6 cuerdas)
    pitches = [48, 52, 55, 60, 64] + [40, 45, 50, 55, 59, 64, 67]
    starts = [0] * 5 + [960] * 7
    pred = m.predict(pitches, starts)
    assert [fb.pitch_of(p) for p in pred] == pitches
    assert len({p.string for p in pred[:5]}) == 5
    pisados = [p.fret for p in pred[:5] if p.fret > 0]
    assert max(pisados) - min(pisados) <= 7
    # el evento imposible recibe el traste más bajo de cada nota
    assert pred[5:] == [min(fb.candidates(p), key=lambda q: q.fret) for p in pitches[5:]]


def test_vacio_y_sin_inicios():
    fb = Fretboard()
    assert AStarBurlet(fb).predict([]) == []
    pred = AStarBurlet(fb).predict([64, 67, 69])
    assert [fb.pitch_of(p) for p in pred] == [64, 67, 69]
