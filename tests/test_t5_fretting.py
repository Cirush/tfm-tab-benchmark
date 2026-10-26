"""Tests del pipeline T5 (sin entrenar: comprueban el funcionamiento, no la calidad)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.fretboard import Fretboard, Position
from src.metrics import pitch_accuracy
from src.models.t5_fretting import (
    WindowDataset, build_model, encode_pair, evaluate_tracks, predict_tracks,
)
from src.tokenizer_v3 import EOS, V3Tokenizer

TOK = V3Tokenizer()
FB = Fretboard()


def _song(n, start=50):
    notes = [(start + k % 15, 480) for k in range(n)]
    return {"notes": notes, "tabs": [(FB.candidates(p)[-1], d) for p, d in notes]}


def _tiny():
    torch.manual_seed(0)
    return build_model(TOK, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=2)


def test_config_del_paper():
    cfg = build_model(TOK).config
    assert (cfg.d_model, cfg.d_ff, cfg.num_layers, cfg.num_decoder_layers, cfg.num_heads) == \
        (128, 1024, 3, 3, 4)


def test_encode_pair_formato():
    src, labels = encode_pair(TOK, _song(5))
    assert len(src) == 3 * 5 + 2                     # <bos> + 3/nota + <eos>
    assert len(labels) == 2 * 5 + 1                  # 2/nota + <eos>, sin <bos>
    assert labels[-1] == TOK.vocab[EOS] and labels[0] != TOK.vocab["<bos>"]


def test_collate_ignora_padding_en_loss():
    ds = WindowDataset([_song(3), _song(6)], TOK)
    src, mask, labels = ds.collate([ds[0], ds[1]])
    assert mask[0].sum() == 3 * 3 + 2 and mask[1].sum() == 3 * 6 + 2
    assert (labels[0, 2 * 3 + 1:] == -100).all()


def test_predict_tracks_una_posicion_por_nota():
    songs = [_song(45), _song(7), _song(20)]         # varios trozos y longitudes
    preds = predict_tracks(_tiny(), TOK, [s["notes"] for s in songs], fb=FB, batch_size=2)
    for p, s in zip(preds, songs):
        for stage in ("raw", "window", "final"):
            assert len(p[stage]) == len(s["notes"])
        assert all(isinstance(x, Position) for x in p["final"])
        assert pitch_accuracy(p["final"], [q for q, _ in s["notes"]], FB) == 1.0


def test_evaluate_tracks_claves():
    res = evaluate_tracks(_tiny(), TOK, [_song(25)])
    assert res["n_notes"] == 25 and res["pitch_acc_final"] == 1.0
    for k in ("tab_acc_raw", "tab_acc_window", "tab_acc_final", "difficulty_pred"):
        assert k in res


def test_predict_tracks_con_contexto_y_anticipacion():
    songs = [_song(70), _song(9)]
    preds = predict_tracks(_tiny(), TOK, [s["notes"] for s in songs], fb=FB,
                           chunk=40, ctx=5, commit=20)
    for p, s in zip(preds, songs):
        assert all(len(p[st]) == len(s["notes"]) for st in ("raw", "window", "final"))
        assert pitch_accuracy(p["final"], [q for q, _ in s["notes"]], FB) == 1.0
