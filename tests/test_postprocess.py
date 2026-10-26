"""Tests del post-procesado (alineamiento + corrección de pitch del paper)."""

import random

from src.fretboard import Fretboard, Position
from src.metrics import pitch_accuracy, tab_accuracy
from src.postprocess import fill_fallback, match_window, postprocess, raw_alignment

FB = Fretboard()
P = Position


def _gt(n, seed=0):
    rng = random.Random(seed)
    tabs = [P(rng.randint(1, 6), rng.randint(0, 12)) for _ in range(n)]
    return tabs, [FB.pitch_of(t) for t in tabs]


def _escala(n=24):
    """Escala cromática: todos los pitches distintos (sin vecinas del mismo pitch)."""
    pitches = list(range(40, 40 + n))
    return [FB.candidates(p)[-1] for p in pitches], pitches


def test_salida_perfecta_no_se_toca():
    gt, pitches = _gt(30)
    assert postprocess(gt, pitches, FB) == gt


def test_nota_saltada_se_realinea():
    gt, pitches = _escala()
    pred = gt[:10] + gt[11:]            # el modelo se salta la nota 10
    raw = raw_alignment(pred, len(pitches))
    assert tab_accuracy(raw, gt) < 0.5  # sin post-procesado, todo desplazado
    out = postprocess(pred, pitches, FB)
    assert len(out) == len(gt)
    assert tab_accuracy(out, gt) >= (len(gt) - 1) / len(gt)  # solo falla la perdida
    assert pitch_accuracy(out, pitches, FB) == 1.0


def test_nota_extra_se_ignora():
    gt, pitches = _escala()
    pred = gt[:5] + [P(6, 20)] + gt[5:]  # TAB espurio insertado
    assert postprocess(pred, pitches, FB) == gt


def test_limitacion_conocida_vecina_mismo_pitch():
    # Mismo pitch (Mi4) en dos posiciones: (1,0) y (2,5). Tras insertar un TAB
    # espurio al principio, para la nota 2 la predicción j=2 es la de la nota 1
    # (1,0), que suena igual -> la versión literal del paper la elige aunque la
    # correcta, (2,5), esté en j=3. Documentado en postprocess.py.
    gt = [P(3, 0), P(1, 0), P(2, 5)]
    pitches = [FB.pitch_of(t) for t in gt]
    pred = [P(6, 20)] + gt
    assert postprocess(pred, pitches, FB) == [P(3, 0), P(1, 0), P(1, 0)]


def test_pitch_incorrecto_usa_fallback():
    pitches = [64, 67]                 # Mi4, Sol4
    pred = [P(1, 0), P(1, 5)]          # la 2ª suena La4: no hay match en la ventana
    aligned = match_window(pred, pitches, FB)
    assert aligned == [P(1, 0), None]
    assert fill_fallback(aligned, pitches, FB) == [P(1, 0), FB.candidates(67)[0]]


def test_ventana_limita_la_busqueda():
    pitches = [40] + [64] * 7
    pred = [P(1, 0)] * 6 + [P(6, 0)] + [P(1, 0)]   # el Mi2 aparece a distancia 6
    assert match_window(pred, pitches, FB, window=5)[0] is None
    assert match_window(pred, pitches, FB, window=6)[0] == P(6, 0)


def test_empate_prefiere_prediccion_anterior():
    pitches = [50, 64, 50]
    pred = [P(4, 0), P(2, 5), P(5, 5)]   # para la nota 1 (Mi4): j=0 no, j=1 sí
    assert match_window(pred, pitches, FB)[1] == P(2, 5)
    # nota 1 sin match exacto en j=1; candidatos a distancia 1 por ambos lados
    pred2 = [P(1, 0), P(3, 0), P(2, 5)]  # j=0 (1,0) y j=2 (2,5) suenan a 64
    assert match_window(pred2, pitches, FB)[1] == P(1, 0)


def test_tab_invalido_no_rompe():
    pitches = [64]
    assert postprocess([P(7, 3), P(0, 1), P(1, 30)], pitches, FB) == [FB.candidates(64)[0]]


def test_siempre_100_pitch_con_salida_aleatoria():
    rng = random.Random(1)
    for _ in range(50):
        gt, pitches = _gt(rng.randint(1, 40), seed=rng.random())
        pred = [P(rng.randint(1, 7), rng.randint(0, 24)) for _ in range(rng.randint(0, 50))]
        out = postprocess(pred, pitches, FB)
        assert len(out) == len(pitches)
        assert pitch_accuracy(out, pitches, FB) == 1.0


def test_raw_alignment_recorta_y_rellena():
    assert raw_alignment([P(1, 0)] * 3, 2) == [P(1, 0)] * 2
    assert raw_alignment([P(1, 0)], 3) == [P(1, 0), None, None]
