"""Tests de la decodificación restringida por pitch (Hsu et al. 2026)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.fretboard import Fretboard, Position
from src.metrics import pitch_accuracy
from src.models import bert_fretting, gpt2_fretting, t5_fretting
from src.models.constrained import allowed_tokens, valid_tab_ids
from src.tokenizer_v3 import EOS, V3Tokenizer

TOK = V3Tokenizer()
FB = Fretboard()


def _notes(n, start=48):
    return [(start + (k * 5) % 24, 480 if k % 2 else 240) for k in range(n)]


def test_tabs_validos_por_pitch():
    ids = valid_tab_ids(TOK, FB, 64)    # Mi4: (1,0) (2,5) (3,9) (4,14) (5,19) (6,24)
    assert sorted(TOK.inv_vocab[i] for i in ids) == sorted(
        f"TAB<{p.string},{p.fret}>" for p in FB.candidates(64))
    assert valid_tab_ids(TOK, FB, 40) == [TOK.vocab["TAB<6,0>"]]   # Mi2: solo 6ª al aire


def test_estructura_forzada():
    notes = [(64, 480), (45, 240)]
    cache = {}
    assert set(allowed_tokens(TOK, FB, notes, 0, cache)) == set(valid_tab_ids(TOK, FB, 64))
    assert allowed_tokens(TOK, FB, notes, 1, cache) == [TOK.vocab["TIME_SHIFT<480>"]]
    assert set(allowed_tokens(TOK, FB, notes, 2, cache)) == set(valid_tab_ids(TOK, FB, 45))
    assert allowed_tokens(TOK, FB, notes, 3, cache) == [TOK.vocab["TIME_SHIFT<240>"]]
    assert TOK.vocab[EOS] in allowed_tokens(TOK, FB, notes, 4, cache)


@pytest.mark.parametrize("arch, kw", [
    (t5_fretting, dict(d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=2)),
    (gpt2_fretting, dict(n_embd=32, n_inner=64, n_layer=1, n_head=2)),
    (bert_fretting, dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64)),
])
def test_modelo_sin_entrenar_acierta_todas_las_notas_sin_postprocesado(arch, kw):
    torch.manual_seed(0)
    model = arch.build_model(TOK, **kw)
    tracks = [_notes(45), _notes(7)]
    preds = arch.predict_tracks(model, TOK, tracks, fb=FB, constrained=True, batch_size=2)
    for p, notes in zip(preds, tracks):
        assert len(p["raw"]) == len(notes) and all(isinstance(x, Position) for x in p["raw"])
        assert pitch_accuracy(p["raw"], [q for q, _ in notes], FB) == 1.0   # sin post-procesar
        assert p["raw"] == p["final"]                                         # el post-procesado sobra
