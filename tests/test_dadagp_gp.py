"""Tests del loader de DadaGP desde GuitarPro (filtro acústico del paper).

Los unitarios no dependen del dataset. Los de integración se saltan si no está
el dataset o el índice (`scripts/build_acoustic_index.py`).
"""

import os
import pytest

from src.fretboard import STANDARD_TUNING, DROP_D_TUNING, Position
from src.data.dadagp_gp import (
    TrackInfo,
    dedup,
    dedup_paper,
    is_guitar_named,
    load_index,
    load_track_poly,
    load_tracks,
    save_tracks,
    song_key,
    split_paper,
    track_notes_poly,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADAGP_ROOT = os.path.join(os.path.dirname(PROJECT), "DadaGP-v1.1")
INDEX = os.path.join(PROJECT, "data", "acoustic_index.json")
needs_index = pytest.mark.skipif(
    not (os.path.isdir(DADAGP_ROOT) and os.path.exists(INDEX)),
    reason="DadaGP o data/acoustic_index.json no presentes",
)


def _info(song, track=1, tuning=STANDARD_TUNING, capo=0, md5=""):
    return TrackInfo(song=song, track=track, program=25, tuning=tuning,
                     capo=capo, artist="a", md5=md5 or song)


# ----------------------- unitarios (sin dataset) --------------------------- #
def test_usable_solo_estandar_sin_cejilla():
    assert _info("a").usable
    assert not _info("a", tuning=DROP_D_TUNING).usable
    assert not _info("a", capo=2).usable
    assert not _info("a", tuning=(64, 59, 55, 50, 45, 40, 35)).usable  # 7 cuerdas


def test_song_key_agrupa_versiones():
    a = song_key("M/Metallica/Metallica - One.gp4")
    assert a == song_key("M/Metallica/Metallica - One (2).gp5")
    assert a == song_key("U/u_lookuperr/Metallica - One.gp3")
    assert a != song_key("M/Metallica/Metallica - Two.gp4")


def test_dedup_elimina_copias_exactas():
    infos = [_info("B/x.gp4", md5="h1"), _info("A/x.gp4", md5="h1"),
             _info("A/x.gp4", track=2, md5="h1"), _info("C/y.gp4", md5="h2")]
    out = dedup(infos)
    assert {i.song for i in out} == {"A/x.gp4", "C/y.gp4"}
    assert len(out) == 3  # se conservan todas las pistas de la copia elegida


def test_save_load_tracks_ida_y_vuelta(tmp_path):
    songs = [{"notes": [(64, 480), (67, 320)],
              "tabs": [(Position(1, 0), 480), (Position(1, 3), 320)],
              "meta": {"song": "x", "track": 1}}]
    path = str(tmp_path / "t.jsonl.gz")
    save_tracks(path, songs)
    assert load_tracks(path) == songs


# ----------------------- integración (con dataset) ------------------------- #
@needs_index
def test_load_track_poly_real_coherente():
    infos, _ = load_index(INDEX)
    for info in [i for i in infos if i.usable and is_guitar_named(i)][:20]:
        s = load_track_poly(DADAGP_ROOT, info)
        if s is None:
            continue
        assert len(s["notes"]) == len(s["tabs"])
        for (pitch, d), (pos, d2) in zip(s["notes"], s["tabs"]):
            assert d == d2 > 0
            assert pitch == STANDARD_TUNING[pos.string - 1] + pos.fret


# ----------------------- réplica del paper ---------------------------------- #
from types import SimpleNamespace as NS

import guitarpro



def _gpnote(string, fret, kind=guitarpro.NoteType.normal):
    return NS(string=string, value=fret, type=kind, realValue=STANDARD_TUNING[string - 1] + fret)


def _beat(start, dur, *notes):
    return NS(start=start, duration=NS(time=dur), notes=list(notes))


def _track(*beats):
    return NS(measures=[NS(voices=[NS(beats=list(beats))])])


def test_filtro_por_nombre_de_pista():
    for name in ("Acoustic Guitar", "Guitarra", "Gitarre 1", "Chitarra", "Violão", "Gtr. 2", "Guitare"):
        assert is_guitar_named(TrackInfo("s", 1, 25, STANDARD_TUNING, 0, "a", name=name)), name
    for name in ("Track 1", "Piste 1", "", "Acoustic", "Voice"):
        assert not is_guitar_named(TrackInfo("s", 1, 25, STANDARD_TUNING, 0, "a", name=name)), name


def test_notas_polifonicas_acorde_y_ligadura():
    track = _track(
        _beat(960, 480, _gpnote(1, 0), _gpnote(3, 2), _gpnote(5, 0)),  # acorde de 3 notas
        _beat(1440, 480, _gpnote(1, 0, guitarpro.NoteType.tie)),      # ligadura: alarga la 1ª
        _beat(1920, 240, _gpnote(2, 1), _gpnote(4, 0, guitarpro.NoteType.dead)),
    )
    rows = track_notes_poly(track)
    # (start, end, pitch, cuerda, traste), orden (start, pitch); la muerta se descarta
    assert rows == [
        (960, 1440, 45, 5, 0),
        (960, 1440, 57, 3, 2),
        (960, 1920, 64, 1, 0),      # 480 + 480 por la ligadura
        (1920, 2160, 60, 2, 1),
    ]


def test_dedup_paper_un_fichero_por_titulo():
    infos = [TrackInfo("B/Art - Song.gp4", 1, 25, STANDARD_TUNING, 0, "a"),
             TrackInfo("A/Art - Song (2).gp5", 1, 25, STANDARD_TUNING, 0, "a"),
             TrackInfo("A/Art - Song (2).gp5", 2, 24, STANDARD_TUNING, 0, "a"),
             TrackInfo("C/Art - Other.gp4", 1, 25, STANDARD_TUNING, 0, "a")]
    out = dedup_paper(infos)
    assert {(i.song, i.track) for i in out} == {("A/Art - Song (2).gp5", 1),
                                                ("A/Art - Song (2).gp5", 2),
                                                ("C/Art - Other.gp4", 1)}


def test_split_paper_estratificado_y_por_fichero():
    infos = [TrackInfo(f"X/s{k}.gp4", t, 25, STANDARD_TUNING, 0, "a", key=key)
             for k, key in enumerate(["CMajor"] * 50 + ["GMajor"] * 30) for t in (1, 2)]
    sp = split_paper(infos, seed=0)
    assert sum(len(v) for v in sp.values()) == len(infos)
    where = {}
    for name, part in sp.items():
        for i in part:
            assert where.setdefault(i.song, name) == name     # pistas de un fichero juntas
    for key, total in (("CMajor", 50), ("GMajor", 30)):         # 80/10/10 dentro de cada tonalidad
        n_train = len({i.song for i in sp["train"] if i.key == key})
        assert n_train == round(total * 0.8)
    assert split_paper(infos, seed=0) == sp
