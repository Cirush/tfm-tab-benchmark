"""Loader de GAPS (Riley et al., ISMIR 2024).

GAPS (Guitar-Aligned Performance Scores): 289 partituras de guitarra clásica en
MusicXML procedentes de ediciones de ClassClef (Guitar Pro), cada una con una
parte de partitura y otra de **tablatura** (cuerda y traste por nota).
Zenodo 13962272, licencia CC BY-NC-SA 4.0 (uso no comercial citando a los
autores; no se versiona).

Aporta repertorio clásico profesional, ausente en DadaGP y en GuitarSet. Se usa
como test adicional y, en el conjunto mezclado (`scripts/build_mixed_dataset.py`),
también para entrenar y validar. Como en el modelo "standard" del paper, solo
piezas en afinación estándar.

Lectura de la parte de tablatura (clave TAB):
- La posición en el tiempo se sigue nota a nota: <chord> no avanza (misma
  posición que la nota anterior), <backup>/<forward> mueven el cursor (varias
  voces en un compás), los silencios avanzan. Ticks = divisiones · 960 / divisions.
- Una nota ligada (<tie type="stop">) no es un ataque nuevo: alarga la nota
  abierta en esa cuerda (como en `dadagp_gp.track_notes_poly`).
- Se descartan las notas de adorno (<grace>), como en el resto del proyecto.
- Se comprueba que afinación[cuerda] + traste == nota escrita; las notas que no
  cuadran (errores de la partitura) se descartan y se cuentan.
Salida en el formato del proyecto: {"notes", "tabs", "starts", "meta"}.
"""

import glob
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from ..fretboard import Position, STANDARD_TUNING

TICKS_PER_QUARTER = 960
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _t(e) -> Optional[str]:
    return e.text.strip() if e is not None and e.text and e.text.strip() else None


def _midi(step: str, octave: str, alter: Optional[str]) -> int:
    return 12 * (int(octave) + 1) + PC[step] + (int(float(alter)) if alter else 0)


def tab_part(root) -> Optional[ET.Element]:
    """La parte cuya clave es TAB (la tablatura)."""
    for part in root.findall("part"):
        if any(_t(c.find("sign")) == "TAB" for c in part.iter("clef")):
            return part
    return None


def tuning_of(part) -> Tuple[int, ...]:
    """Afinación declarada, de la cuerda 1 (aguda, línea 6) a la 6 (grave, línea 1)."""
    lines = {}
    for st in part.iter("staff-tuning"):
        lines[int(st.get("line"))] = _midi(_t(st.find("tuning-step")), _t(st.find("tuning-octave")),
                                           _t(st.find("tuning-alter")))
    return tuple(lines[k] for k in sorted(lines, reverse=True))


def parse_tab(part) -> Tuple[List[Tuple[int, int, int, int, int]], Dict[str, int]]:
    """[(start, end, pitch, cuerda, traste)] de la parte de tablatura y contadores."""
    divisions = 1
    measure_start = 0          # en divisiones
    notes: List[list] = []
    open_by_string: Dict[int, list] = {}
    stats = {"incoherentes": 0, "adorno": 0}
    for measure in part.findall("measure"):
        cursor = 0
        last_onset = 0
        measure_len = 0
        for el in measure:
            if el.tag == "attributes" and _t(el.find("divisions")):
                divisions = int(_t(el.find("divisions")))
            elif el.tag == "backup":
                cursor -= int(_t(el.find("duration")))
            elif el.tag == "forward":
                cursor += int(_t(el.find("duration")))
            elif el.tag == "note":
                if el.find("grace") is not None:
                    stats["adorno"] += 1
                    continue
                dur = int(_t(el.find("duration")) or 0)
                if el.find("chord") is not None:
                    onset = last_onset
                else:
                    onset = cursor
                    cursor += dur
                last_onset = onset
                measure_len = max(measure_len, cursor)
                if el.find("rest") is not None:
                    continue
                tech = el.find("notations/technical")
                if tech is None or not _t(tech.find("string")) or not _t(tech.find("fret")):
                    continue
                s, f = int(_t(tech.find("string"))), int(_t(tech.find("fret")))
                to_ticks = lambda d: round((measure_start + d) * TICKS_PER_QUARTER / divisions)
                start, end = to_ticks(onset), to_ticks(onset + dur)
                tie_stop = any(t.get("type") == "stop" for t in el.findall("tie"))
                if tie_stop and s in open_by_string:
                    open_by_string[s][1] = max(open_by_string[s][1], end)
                    continue
                p = el.find("pitch")
                try:
                    pitch = _midi(_t(p.find("step")), _t(p.find("octave")), _t(p.find("alter")))
                except (AttributeError, KeyError, TypeError, ValueError):
                    stats["incoherentes"] += 1
                    continue
                rec = [start, end, pitch, s, f]
                notes.append(rec)
                open_by_string[s] = rec
        measure_start += measure_len
    notes.sort(key=lambda r: (r[0], r[2]))
    return [tuple(r) for r in notes], stats


def load_score(path: str, min_notes: int = 8, standard_only: bool = True) -> Optional[dict]:
    """Una partitura de GAPS en el formato del proyecto (None si no pasa filtros)."""
    root = ET.parse(path).getroot()
    part = tab_part(root)
    if part is None:
        return None
    tuning = tuning_of(part)
    if standard_only and tuning != STANDARD_TUNING:
        return None
    rows, stats = parse_tab(part)
    good = [r for r in rows if 1 <= r[3] <= 6 and 0 <= r[4] <= 24
            and tuning[r[3] - 1] + r[4] == r[2]]
    stats["incoherentes"] += len(rows) - len(good)
    if len(good) < min_notes:
        return None
    name = os.path.basename(path)[: -len(".xml")]
    return {
        "notes": [(pitch, max(1, end - start)) for start, end, pitch, _, _ in good],
        "tabs": [(Position(s, f), max(1, end - start)) for start, end, _, s, f in good],
        "starts": [start for start, _, _, _, _ in good],
        "meta": {"song": name, "tuning": list(tuning), "n_notes": len(good), **stats},
    }


def load_gaps(root: str, **kw) -> List[dict]:
    """Todas las partituras de GAPS que pasan los filtros (orden alfabético)."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "musicxml", "*.xml"))):
        song = load_score(path, **kw)
        if song is not None:
            out.append(song)
    return out


# --------------------------------------------------------------------------- #
# Control de fugas: obras de GAPS cuyo título aparece en el train de DadaGP
# --------------------------------------------------------------------------- #
import csv
import re
import unicodedata

# Palabras demasiado genéricas para identificar una obra por sí solas
TITLE_STOP = set("the and from with for minor major opus prelude etude study sonata suite allegro "
                 "andante moderato waltz dance song theme variations part movement guitar piece".split())


def _ascii(x: str) -> str:
    return unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()


def title_words(x: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", _ascii(x)) if w not in TITLE_STOP}


def overlapping_scores(gaps_root: str, train_song_names, ids=None) -> Dict[str, Tuple[str, List[str]]]:
    """{id GAPS: (título, [pistas de train con ese título])}.

    Criterio conservador: todas las palabras significativas del título de la obra
    de GAPS aparecen en el nombre de una pista de entrenamiento (títulos de una
    sola palabra, solo si tiene ≥ 6 letras). Incluye coincidencias de nombre
    genérico; se prefiere excluir de más a tener fuga.
    """
    with open(os.path.join(gaps_root, "gaps_v1_metadata.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    train = [(n, title_words(n.split(" - ", 1)[-1])) for n in sorted(set(train_song_names))]
    out = {}
    for r in rows:
        sid = (r.get("scorehash") or "").strip()
        if ids is not None and sid not in ids:
            continue
        w = title_words(r.get("title", ""))
        if not w or (len(w) == 1 and len(next(iter(w))) < 6):
            continue
        hits = [n for n, ws in train if w <= ws]
        if hits:
            out[sid] = (r.get("title", ""), hits[:3])
    return out
