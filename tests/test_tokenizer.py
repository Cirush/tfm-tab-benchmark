"""Tests del tokenizador v3 y del generador sintético."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tokenizer_v3 import V3Tokenizer, UNK
from src.fretboard import Position, Fretboard
from src.data.synthetic import make_dataset, train_val_test_split


def test_encode_input_format():
    tok = V3Tokenizer()
    tokens = tok.encode_input([(64, 480), (67, 960)])
    assert tokens == ["NOTE_ON<64>", "TIME_SHIFT<480>", "NOTE_OFF<64>",
                      "NOTE_ON<67>", "TIME_SHIFT<960>", "NOTE_OFF<67>"]

def test_encode_target_format():
    tok = V3Tokenizer()
    tokens = tok.encode_target([(Position(1, 0), 480), (Position(2, 5), 960)])
    assert tokens == ["TAB<1,0>", "TIME_SHIFT<480>", "TAB<2,5>", "TIME_SHIFT<960>"]

def test_decode_target_roundtrip():
    tok = V3Tokenizer()
    tabs = [(Position(3, 9), 240), (Position(6, 0), 1920)]
    assert tok.decode_target(tok.encode_target(tabs)) == [p for p, _ in tabs]

def test_decode_tolerates_noise():
    """El decode debe sobrevivir a salidas ruidosas del modelo."""
    tok = V3Tokenizer()
    noisy = ["TAB<1,0>", "<unk>", "TIME_SHIFT<480>", "TAB<malo>", "TAB<2,5>"]
    assert tok.decode_target(noisy) == [Position(1, 0), Position(2, 5)]

def test_quantize_snaps_to_nearest():
    tok = V3Tokenizer()
    assert tok.quantize(500) == 480
    assert tok.quantize(100) == 120
    assert tok.quantize(5000) == 3840

def test_ids_roundtrip_and_unk():
    tok = V3Tokenizer()
    tokens = ["NOTE_ON<64>", "TIME_SHIFT<480>", "NOTE_OFF<64>"]
    ids = tok.to_ids(tokens)
    assert tok.to_tokens(ids) == tokens          # bos/eos filtrados
    assert tok.to_ids(["INVENTADO"], add_special=False) == [tok.vocab[UNK]]

def test_vocab_is_reasonable():
    tok = V3Tokenizer()
    # 4 especiales + 71*2 note on/off + 8 shifts + 7*25 tabs = 329
    assert tok.vocab_size == 4 + 71 * 2 + 8 + 175

def test_synthetic_gt_is_valid():
    """Todo GT sintético debe producir el pitch correcto en el mástil."""
    fb = Fretboard()
    data = make_dataset(20, seed=1)
    for song in data:
        for (pitch, _), (pos, _) in zip(song["notes"], song["tabs"]):
            assert fb.pitch_of(pos) == pitch

def test_split_no_overlap():
    data = make_dataset(50, seed=2)
    train, val, test = train_val_test_split(data, seed=2)
    assert len(train) + len(val) + len(test) == 50
    ids = lambda subset: {id(s) for s in subset}
    assert not (ids(train) & ids(val)) and not (ids(train) & ids(test))


def test_vocabulario_de_duraciones_desde_datos(tmp_path):
    from src.tokenizer_v3 import V3Tokenizer
    tok = V3Tokenizer.from_data([480, 480, 240, 320, 7], min_count=1)
    assert tok.durations == (7, 240, 320, 480)
    assert "TIME_SHIFT<320>" in tok.vocab
    assert V3Tokenizer.from_data([480, 480, 240], min_count=2).durations == (480,)
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    tok2 = V3Tokenizer.load(path)
    assert tok2.vocab == tok.vocab                 # misma asignación de ids
    assert tok2.quantize(330) == 320               # no vistas -> la más cercana
