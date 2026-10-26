"""Tests del loader de GAPS (MusicXML con tablatura) sin necesidad del dataset."""

import os

import pytest

from src.data.gaps import load_score, overlapping_scores, title_words
from src.fretboard import Position

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAPS_ROOT = os.path.join(os.path.dirname(PROJECT), "GAPS-v1")

TUNING_STD = [("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)]
TUNING_DROPD = [("D", 2)] + TUNING_STD[1:]


def _note(step, octv, string, fret, dur, chord=False, tie=None, grace=False):
    return (f"<note>{'<grace/>' if grace else ''}{'<chord/>' if chord else ''}"
            f"<pitch><step>{step}</step><octave>{octv}</octave></pitch>"
            f"<duration>{dur}</duration>{f'<tie type={chr(34)}{tie}{chr(34)}/>' if tie else ''}"
            f"<notations><technical><string>{string}</string><fret>{fret}</fret></technical></notations></note>")


def _score(tmp_path, measures, tuning=TUNING_STD, name="xx1wc"):
    tun = "".join(f'<staff-tuning line="{i + 1}"><tuning-step>{s}</tuning-step>'
                  f'<tuning-octave>{o}</tuning-octave></staff-tuning>' for i, (s, o) in enumerate(tuning))
    attrs = (f"<attributes><divisions>2</divisions><clef><sign>TAB</sign></clef>"
             f"<staff-details><staff-lines>6</staff-lines>{tun}</staff-details></attributes>")
    body = "".join(f"<measure>{attrs if i == 0 else ''}{m}</measure>" for i, m in enumerate(measures))
    xml = (f"<score-partwise><part id='P1'><measure><note><rest/><duration>8</duration></note></measure></part>"
           f"<part id='P2'>{body}</part></score-partwise>")
    p = tmp_path / f"{name}.xml"
    p.write_text(xml)
    return str(p)


def test_acorde_voces_ligadura_y_adorno(tmp_path):
    # divisions=2 → una negra = 2 divisiones = 960 ticks
    m1 = (_note("E", 4, 1, 0, 2) + _note("C", 3, 5, 3, 2, chord=True)       # acorde en t=0
          + _note("G", 4, 1, 3, 2, tie="start")                             # t=960
          + "<backup><duration>4</duration></backup>"
          + _note("E", 2, 6, 0, 4))                                         # 2ª voz en t=0
    m2 = (_note("A", 4, 1, 5, 2, grace=True)                                # adorno: se descarta
          + _note("G", 4, 1, 3, 2, tie="stop")                              # ligada: alarga la anterior
          + _note("B", 3, 2, 0, 2))
    s = load_score(_score(tmp_path, [m1, m2]), min_notes=1)
    got = list(zip(s["starts"], [p for p, _ in s["notes"]], [t for t, _ in s["tabs"]], [d for _, d in s["notes"]]))
    assert got == [
        (0, 40, Position(6, 0), 1920),     # 2ª voz, redonda... de 4 divisiones = 1920
        (0, 48, Position(5, 3), 960),
        (0, 64, Position(1, 0), 960),
        (960, 67, Position(1, 3), 1920),   # 960 + 960 por la ligadura
        (2880, 59, Position(2, 0), 960),
    ]
    assert s["meta"]["adorno"] == 1


def test_afinacion_no_estandar_se_excluye(tmp_path):
    path = _score(tmp_path, [_note("D", 2, 6, 0, 2) * 9], tuning=TUNING_DROPD)
    assert load_score(path, min_notes=1) is None
    assert load_score(path, min_notes=1, standard_only=False) is not None


def test_nota_incoherente_se_descarta(tmp_path):
    notas = _note("E", 4, 1, 0, 2) * 8 + _note("F", 4, 1, 0, 2)   # Fa4 en 1ª al aire: incoherente
    s = load_score(_score(tmp_path, [notas]), min_notes=1)
    assert len(s["notes"]) == 8 and s["meta"]["incoherentes"] == 1


def test_titulos_y_control_de_fugas(tmp_path):
    assert title_words("BWV 565 Toccata & Fugue") == {"toccata", "fugue"}
    (tmp_path / "gaps_v1_metadata.csv").write_text(
        "scorehash,title\nAAA,Toccata and Fugue\nBBB,Preludio\nCCC,Recuerdos de la Alhambra\n")
    out = overlapping_scores(str(tmp_path), ["Bach - Toccata And Fugue (Bwv 565).gp4", "Otro - Nada.gp3"])
    assert set(out) == {"AAA"}            # "Preludio" (genérico, < 6 letras útiles) no cuenta


@pytest.mark.skipif(not os.path.isdir(GAPS_ROOT), reason="GAPS no descargado")
def test_gaps_real_coherente():
    from src.data.gaps import load_gaps
    from src.fretboard import STANDARD_TUNING
    songs = load_gaps(GAPS_ROOT)
    assert len(songs) > 200
    for s in songs[:20]:
        for (p, d), (pos, _) in zip(s["notes"], s["tabs"]):
            assert d > 0 and p == STANDARD_TUNING[pos.string - 1] + pos.fret
