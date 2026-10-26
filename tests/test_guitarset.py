"""Tests del loader de GuitarSet (sin necesidad del dataset real)."""

import json
import os

import pytest

from src.data.guitarset import load_guitarset, load_jams, parse_name
from src.fretboard import Position

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS_ROOT = os.path.join(os.path.dirname(PROJECT), "GuitarSet-1.1")


def _jams(tmp_path, notes_by_source, tempo=120.0, name="00_Jazz1-120-C_solo"):
    anns = [{"namespace": "tempo", "data": [{"time": 0, "duration": 10, "value": tempo}],
             "annotation_metadata": {"data_source": ""}}]
    for source, notes in notes_by_source.items():
        anns.append({"namespace": "note_midi", "annotation_metadata": {"data_source": str(source)},
                     "data": [{"time": t, "duration": d, "value": v, "confidence": None}
                              for t, d, v in notes]})
    p = tmp_path / f"{name}.jams"
    p.write_text(json.dumps({"annotations": anns, "file_metadata": {"title": name}}))
    return str(p)


def test_parse_name():
    assert parse_name("03_SS3-84-Bb_comp") == {"player": "03", "style": "SS", "tempo": 84,
                                              "key": "Bb", "part": "comp"}


def test_cuerdas_trastes_y_ticks(tmp_path):
    # tempo 120 -> 1 s = 2 negras = 1920 ticks
    path = _jams(tmp_path, {
        0: [(0.0, 0.5, 43.1)],          # Mi grave (40) -> traste 3, cuerda 6
        5: [(0.0, 0.25, 66.9), (1.0, 0.5, 64.0)],   # Mi agudo: trastes 3 y 0, cuerda 1
    })
    s = load_jams(path, min_notes=1)
    assert s["notes"] == [(43, 960), (67, 480), (64, 960)]      # orden (inicio, pitch)
    assert [t for t, _ in s["tabs"]] == [Position(6, 3), Position(1, 3), Position(1, 0)]
    assert s["meta"]["style"] == "Jazz" and s["meta"]["part"] == "solo"


def test_min_notes(tmp_path):
    assert load_jams(_jams(tmp_path, {5: [(0.0, 0.5, 64.0)]}), min_notes=8) is None


@pytest.mark.skipif(not os.path.isdir(GS_ROOT), reason="GuitarSet no descargado")
def test_guitarset_real_completo_y_coherente():
    songs = load_guitarset(GS_ROOT)
    assert len(songs) == 360
    from src.fretboard import STANDARD_TUNING
    for s in songs[:40]:
        for (p, d), (pos, d2) in zip(s["notes"], s["tabs"]):
            assert d == d2 > 0 and p == STANDARD_TUNING[pos.string - 1] + pos.fret


def _frag(name, n=3):
    return {"notes": [(64, 470)] * n, "tabs": [(Position(1, 0), 470)] * n,
            "meta": {"song": name, **parse_name(name)}}


def test_split_por_progresion_sin_fuga_y_por_estilo():
    from src.data.guitarset import progression, split_guitarset, STYLES
    songs = [_frag(f"0{p}_{st}{k}-{t}-C_{part}") for st in STYLES for k in (1, 2, 3)
             for t in (100, 120) for p in range(6) for part in ("comp", "solo")]
    train, test = split_guitarset(songs, seed=0)
    assert len(train) + len(test) == len(songs)
    assert {progression(s) for s in train}.isdisjoint({progression(s) for s in test})
    assert sorted(s["meta"]["style"] for s in test[::24]) == sorted(STYLES)  # 1 progresión por estilo
    assert len(test) == len(songs) // 3
    assert split_guitarset(songs, seed=0) == (train, test)


def test_snap_durations():
    from src.data.guitarset import snap_durations
    s = snap_durations([_frag("00_Jazz1-120-C_solo")], [240, 480, 960])[0]
    assert s["notes"][0] == (64, 480) and s["tabs"][0][1] == 480
