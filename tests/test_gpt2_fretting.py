"""Tests del pipeline GPT2 (sin entrenar: comprueban el funcionamiento, no la calidad)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.fretboard import Fretboard
from src.metrics import pitch_accuracy
from src.models.gpt2_fretting import (
    WindowDataset, build_model, encode_pair, predict_tracks, prompt_ids, special_ids,
)
from src.tokenizer_v3 import EOS, V3Tokenizer

TOK = V3Tokenizer()
FB = Fretboard()


def _song(n, start=50):
    notes = [(start + k % 15, 480) for k in range(n)]
    return {"notes": notes, "tabs": [(FB.candidates(p)[-1], d) for p, d in notes]}


def _tiny():
    torch.manual_seed(0)
    return build_model(TOK, n_embd=32, n_inner=64, n_layer=1, n_head=2)


def test_config_equivalente_al_t5():
    n = sum(p.numel() for p in build_model(TOK).parameters())
    assert 1.5e6 < n < 4e6     # mismo orden que el T5 del paper (2,8M)


def test_secuencia_y_labels():
    midi, tabs = special_ids(TOK)
    seq, labels = encode_pair(TOK, _song(4))
    assert seq[0] == midi and seq[1 + 3 * 4] == tabs              # MIDI: 12 tokens TABS:
    assert len(seq) == 1 + 3 * 4 + 1 + 2 * 4 + 1 and seq[-1] == TOK.vocab[EOS]
    assert labels[: 1 + 3 * 4 + 1] == [-100] * 14                 # el prompt no se predice
    assert labels[14:] == seq[14:]


def test_prompt_con_prefijo_de_contexto():
    s = _song(3)
    p = prompt_ids(TOK, s["notes"], prefix=s["tabs"][:1])
    assert len(p) == 1 + 9 + 1 + 2


def test_generacion_por_lotes_igual_que_individual():
    # el relleno a la izquierda no debe cambiar el resultado de un prompt
    m = _tiny()
    short, long = _song(7), _song(20)
    solo = predict_tracks(m, TOK, [short["notes"]], fb=FB)[0]["raw"]
    juntos = predict_tracks(m, TOK, [long["notes"], short["notes"]], fb=FB)[1]["raw"]
    assert solo == juntos


def test_una_posicion_por_nota_y_pitch_100():
    songs = [_song(45), _song(9)]
    preds = predict_tracks(_tiny(), TOK, [s["notes"] for s in songs], fb=FB, batch_size=1)
    for p, s in zip(preds, songs):
        assert all(len(p[k]) == len(s["notes"]) for k in ("raw", "window", "final"))
        assert pitch_accuracy(p["final"], [q for q, _ in s["notes"]], FB) == 1.0


def test_collate():
    ds = WindowDataset([_song(3), _song(5)], TOK)
    ids, mask, labels = ds.collate([ds[0], ds[1]])
    assert mask[0].sum() == len(ds[0][0]) and (labels[0, len(ds[0][0]):] == -100).all()
