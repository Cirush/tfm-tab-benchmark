"""Tests del pipeline BERT (sin entrenar: comprueban el funcionamiento, no la calidad)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.fretboard import Fretboard, Position
from src.metrics import pitch_accuracy
from src.models.bert_fretting import (
    WindowDataset, build_model, encode, mask_id, predict_tracks,
)
from src.tokenizer_v3 import V3Tokenizer

TOK = V3Tokenizer()
FB = Fretboard()


def _song(n, start=50):
    notes = [(start + k % 15, 480) for k in range(n)]
    return {"notes": notes, "tabs": [(FB.candidates(p)[-1], d) for p, d in notes]}


def _tiny():
    torch.manual_seed(0)
    return build_model(TOK, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=64)


def test_config_de_midi_to_tab():
    c = build_model(TOK).config
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.intermediate_size) == \
        (384, 6, 6, 1536)


def test_tab_tras_note_on_y_enmascarado():
    s = _song(3)
    ids, labels = encode(TOK, s["notes"], [t for t, _ in s["tabs"]])
    assert len(ids) == 1 + 4 * 3 + 1
    m = mask_id(TOK)
    for k, (pos, _) in enumerate(s["tabs"]):
        assert TOK.inv_vocab[ids[1 + 4 * k]].startswith("NOTE_ON")
        assert ids[2 + 4 * k] == m                                    # TAB enmascarado
        assert labels[2 + 4 * k] == TOK.vocab[f"TAB<{pos.string},{pos.fret}>"]
    assert sum(l != -100 for l in labels) == 3                        # loss solo en los TAB


def test_contexto_visible_sin_loss():
    s = _song(3)
    ids, labels = encode(TOK, s["notes"], [t for t, _ in s["tabs"]], n_visible=1)
    assert TOK.inv_vocab[ids[2]].startswith("TAB<") and labels[2] == -100
    assert sum(l != -100 for l in labels) == 2


def test_una_posicion_por_nota_y_pitch_100():
    songs = [_song(45), _song(9)]
    preds = predict_tracks(_tiny(), TOK, [s["notes"] for s in songs], fb=FB, batch_size=1)
    for p, s in zip(preds, songs):
        assert all(len(p[k]) == len(s["notes"]) for k in ("raw", "window", "final"))
        assert all(isinstance(x, Position) for x in p["raw"])       # nunca le faltan TABs
        assert pitch_accuracy(p["final"], [q for q, _ in s["notes"]], FB) == 1.0


def test_dataset_mitad_con_contexto():
    ds = WindowDataset([_song(5)] * 200, TOK, seed=0)
    con = sum(TOK.inv_vocab.get(x[2], "").startswith("TAB<") for x, _ in ds.items)
    assert 70 < con < 130
