"""Tests de verificación de la lógica base del benchmark."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.fretboard import Fretboard, Position
from src.metrics import tab_accuracy, string_accuracy, difficulty, difficulty_score, evaluate
from src.models.baseline_minfret import MinFretBaseline
from src.models.astar import AStarFretting


# ---------- Fretboard ----------

def test_e4_candidates():
    """Mi4 (64) debe tener 6 posiciones en afinación estándar con 24 trastes."""
    fb = Fretboard()
    cands = fb.candidates(64)
    expected = [Position(1, 0), Position(2, 5), Position(3, 9),
                Position(4, 14), Position(5, 19), Position(6, 24)]
    assert cands == expected  # 6 con 24 trastes (la 6ª llega justo en el 24)

def test_low_e_single_candidate():
    """Mi2 (40) solo puede tocarse en la 6ª al aire."""
    fb = Fretboard()
    assert fb.candidates(40) == [Position(6, 0)]

def test_out_of_range():
    fb = Fretboard()
    assert fb.candidates(39) == []      # por debajo del Mi grave
    assert fb.candidates(64 + 25) == [] # por encima del traste 24 en 1ª

def test_pitch_roundtrip():
    fb = Fretboard()
    for pitch in range(40, 88):
        for pos in fb.candidates(pitch):
            assert fb.pitch_of(pos) == pitch


# ---------- Baseline ----------

def test_baseline_prefers_open_strings():
    model = MinFretBaseline()
    # Mi4: la opción de menor traste es 1ª al aire
    assert model.predict([64]) == [Position(1, 0)]

def test_baseline_no_context():
    """El baseline decide nota a nota: dos Mi4 seguidos dan la misma posición."""
    model = MinFretBaseline()
    assert model.predict([64, 64]) == [Position(1, 0), Position(1, 0)]


# ---------- A* ----------

def test_astar_valid_pitches():
    """Toda posición devuelta debe producir el pitch pedido."""
    fb = Fretboard()
    model = AStarFretting(fb)
    melody = [40, 45, 50, 55, 59, 64, 67, 72]
    pred = model.predict(melody)
    assert [fb.pitch_of(p) for p in pred] == melody

def test_astar_smoother_than_baseline_on_high_melody():
    """En una melodía aguda en zona alta, A* debe dar digitación no peor
    en difficulty score que el baseline nota-a-nota."""
    melody = [76, 79, 81, 79, 76, 74, 76]  # frase aguda
    base = MinFretBaseline().predict(melody)
    astar = AStarFretting().predict(melody)
    assert difficulty_score(astar) <= difficulty_score(base)

def test_astar_empty():
    assert AStarFretting().predict([]) == []


# ---------- Métricas ----------

def test_tab_accuracy_perfect_and_zero():
    gt = [Position(1, 0), Position(2, 5)]
    assert tab_accuracy(gt, gt) == 1.0
    pred = [Position(3, 9), Position(4, 14)]
    assert tab_accuracy(pred, gt) == 0.0

def test_string_accuracy_partial():
    gt = [Position(1, 0), Position(2, 5)]
    pred = [Position(1, 12), Position(2, 5)]  # 1ª nota: cuerda bien, traste mal
    assert string_accuracy(pred, gt) == 1.0
    assert tab_accuracy(pred, gt) == 0.5

def test_metrics_length_mismatch():
    with pytest.raises(ValueError):
        tab_accuracy([Position(1, 0)], [])

def test_evaluate_keys():
    gt = [Position(1, 0), Position(1, 2)]
    res = evaluate(gt, gt)
    assert res["tab_accuracy"] == 1.0
    assert res["n_notes"] == 2


# ---------- Difficulty score (paper) ----------

def test_difficulty_ecuaciones():
    P = Position
    assert difficulty(P(1, 0), P(1, 0)) == 0.25              # mínimo: solo vertical
    assert difficulty(P(1, 2), P(1, 5)) == 0.5 * 3 + 0.25 * 7 + 0.25   # subir
    assert difficulty(P(1, 5), P(1, 2)) == 0.75 * 3 + 0.25 * 7 + 0.25  # bajar cuesta más
    assert difficulty(P(6, 0), P(1, 0)) == 0.50               # salto de cuerda (|Δ|)
    assert difficulty(P(1, 0), P(6, 0)) == 0.50

def test_difficulty_rango_del_paper():
    P = Position
    assert difficulty(P(6, 0), P(1, 24)) == 18.5   # el máximo que da el paper (subiendo)
    assert difficulty(P(1, 24), P(6, 0)) == 24.5   # bajando es mayor -> errata del rango

def test_difficulty_score_media():
    P = Position
    seq = [P(1, 0), P(1, 0), P(1, 2)]
    assert difficulty_score(seq) == (0.25 + (0.5 * 2 + 0.25 * 2 + 0.25)) / 2
    assert difficulty_score(seq[:1]) == 0.0


# ---------- A* con acordes ----------

from src.models.astar import AStarChords


def test_astar_acordes_agrupa_por_inicio():
    m = AStarChords(onset_tol=120)
    assert m.groups([0, 0, 10, 500, 600, 2000]) == [[0, 1, 2], [3, 4], [5]]


def test_astar_acordes_cuerdas_distintas_y_pitch_correcto():
    fb = Fretboard()
    m = AStarChords(fb)
    # Do mayor abierto (C3 E3 G3 C4 E4) como acorde + una melodía
    pitches = [48, 52, 55, 60, 64, 67, 65, 64]
    starts = [0, 0, 0, 0, 0, 960, 1440, 1920]
    pred = m.predict(pitches, starts)
    assert [fb.pitch_of(p) for p in pred] == pitches
    assert len({p.string for p in pred[:5]}) == 5          # una nota por cuerda en el acorde
    assert max(p.fret for p in pred[:5]) <= 5               # acorde en primera posición


def test_astar_acordes_sin_inicios_funciona_nota_a_nota():
    fb = Fretboard()
    pred = AStarChords(fb).predict([64, 67, 69])
    assert [fb.pitch_of(p) for p in pred] == [64, 67, 69]
