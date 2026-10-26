"""Tests del troceo de pistas (entrenamiento e inferencia por chunks)."""

from src.fretboard import Fretboard, Position
from src.tokenizer_v3 import V3Tokenizer
import random

from src.data.chunking import (
    chunk_plan,
    chunked_predict,
    max_notes_for,
    random_windows,
    training_windows,
)


def _song(n):
    fb = Fretboard()
    notes = [(52 + k % 12, 480) for k in range(n)]
    tabs = [(fb.candidates(p)[0], d) for p, d in notes]
    return {"notes": notes, "tabs": tabs, "meta": {"song": "x"}}


def test_max_notes_512_tokens():
    assert max_notes_for(512) == 170
    # la entrada de 170 notas + <bos>/<eos> cabe en 512 tokens; 171 no
    tok = V3Tokenizer()
    s = _song(171)
    assert len(tok.to_ids(tok.encode_input(s["notes"][:170]))) <= 512
    assert len(tok.to_ids(tok.encode_input(s["notes"]))) > 512


def test_ventanas_cubren_todo_sin_solape_y_alineadas():
    s = _song(400)
    ws = training_windows(s, max_notes=170)
    assert [len(w["notes"]) for w in ws] == [170, 170, 60]
    assert [w["meta"]["start"] for w in ws] == [0, 170, 340]
    # concatenadas reproducen la pista exacta (nada se pierde ni se repite)
    assert sum((w["notes"] for w in ws), []) == s["notes"]
    assert sum((w["tabs"] for w in ws), []) == s["tabs"]
    for w in ws:
        assert len(w["notes"]) == len(w["tabs"])
        assert w["meta"]["song"] == "x"


def test_ventana_final_corta_se_descarta_con_min_notes():
    ws = training_windows(_song(175), max_notes=170, min_notes=8)
    assert [len(w["notes"]) for w in ws] == [170]


def test_chunked_predict_esquema_del_paper():
    s = _song(45)
    calls = []

    def fake(window, prefix):
        calls.append((len(window), list(prefix)))
        # devuelve una posición "marcada" por nota tras el prefijo
        return [Position(1, len(calls)) for _ in window[len(prefix):]]

    out = chunked_predict(s["notes"], fake, chunk=20)
    assert len(out) == 45                      # una posición por nota
    # 1º trozo: 20 notas sin contexto; siguientes: 1 de contexto + trozo
    assert [c[0] for c in calls] == [20, 21, 6]
    assert calls[0][1] == []
    # el prefijo del decoder es el TAB ya predicho para la nota de contexto
    assert calls[1][1] == [(Position(1, 1), s["notes"][19][1])]
    assert calls[2][1] == [(Position(1, 2), s["notes"][39][1])]


def test_random_windows_cubren_todo_y_respetan_rango():
    s = _song(500)
    ws = random_windows(s, 8, 64, random.Random(0))
    assert sum((w["notes"] for w in ws), []) == s["notes"]
    assert sum((w["tabs"] for w in ws), []) == s["tabs"]
    assert all(8 <= len(w["notes"]) <= 64 for w in ws[:-1])
    assert len({len(w["notes"]) for w in ws}) > 3  # longitudes variadas


def test_chunk_plan_paper():
    assert chunk_plan(45) == [(0, 0, 20, 20), (19, 20, 40, 40), (39, 40, 45, 45)]


def test_chunk_plan_con_anticipacion_y_contexto():
    # ve 40 notas nuevas, da por buenas 20, repite 5 de contexto
    plan = chunk_plan(70, chunk=40, ctx=5, commit=20)
    assert plan == [(0, 0, 40, 20), (15, 20, 60, 40), (35, 40, 70, 70)]
    # cada nota se da por buena exactamente una vez
    assert [i for _, st, _, ce in plan for i in range(st, ce)] == list(range(70))


def test_chunked_predict_con_anticipacion():
    s = _song(70)
    calls = []

    def fake(window, prefix):
        calls.append((len(window), [p for p, _ in prefix]))
        return [Position(1, len(calls)) for _ in window[len(prefix):]]

    out = chunked_predict(s["notes"], fake, chunk=40, ctx=5, commit=20)
    assert len(out) == 70
    assert [c[0] for c in calls] == [40, 45, 35]
    # el prefijo son las posiciones YA decididas (del trozo anterior), no las descartadas
    assert calls[1][1] == [Position(1, 1)] * 5
    assert out[:20] == [Position(1, 1)] * 20 and out[20:40] == [Position(1, 2)] * 20
