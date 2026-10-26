"""Loader de GuitarSet (Xi et al., ISMIR 2018).

GuitarSet: 360 fragmentos de guitarra acústica (5 estilos: bossa nova, funk,
jazz, rock, cantautor; acompañamiento y solo; 6 guitarristas) con cuerda y
traste anotados mediante pastilla hexafónica (una señal por cuerda) → digitación
real, no escrita por usuarios de una web. Licencia CC BY 4.0 (se cita; no se
versiona). DOI 10.5281/zenodo.3371780.

Complementa a DadaGP con estilos donde el baseline falla más (jazz, funk) y
sustituye en parte a Leduc (no disponible). Se reparte por progresión de acordes
(`split_guitarset`) para que la misma música no esté en train y test.

Formato JAMS: 6 anotaciones `note_midi`, una por cuerda (`data_source` "0" =
Mi grave ... "5" = Mi agudo), con tiempos en segundos y pitch continuo. Salida
en el formato del proyecto: {"notes": [(pitch, dur)], "tabs": [(Position, dur)],
"meta"}, con tiempos convertidos a ticks (960 por negra) según el tempo anotado,
notas ordenadas por (inicio, pitch) como en `dadagp_gp.track_notes_poly`.
"""

import glob
import json
import os
from typing import List, Optional

from ..fretboard import Position, STANDARD_TUNING

TICKS_PER_QUARTER = 960
# data_source de GuitarSet (0 = Mi grave) -> nuestra cuerda (6 = Mi grave)
OPEN_BY_SOURCE = tuple(reversed(STANDARD_TUNING))   # (40, 45, 50, 55, 59, 64)
STYLES = {"BN": "bossa nova", "Funk": "funk", "Jazz": "jazz", "Rock": "rock",
          "SS": "cantautor"}


def parse_name(name: str) -> dict:
    """`00_BN1-129-Eb_comp` -> jugador, estilo, tempo, tonalidad, parte."""
    player, rest, part = name.split("_")
    style_prog, tempo, key = rest.split("-")
    style = style_prog.rstrip("0123456789")
    return {"player": player, "style": style, "tempo": int(tempo), "key": key, "part": part}


def load_jams(path: str, min_notes: int = 8) -> Optional[dict]:
    """Un fragmento de GuitarSet en el formato del proyecto (None si < min_notes)."""
    with open(path, encoding="utf-8") as fh:
        jams = json.load(fh)
    tempo = next(a["data"][0]["value"] for a in jams["annotations"] if a["namespace"] == "tempo")
    to_ticks = lambda sec: round(sec * tempo / 60 * TICKS_PER_QUARTER)
    rows = []
    for ann in jams["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        source = int(ann["annotation_metadata"]["data_source"])
        string = 6 - source
        for note in ann["data"]:
            pitch = round(note["value"])
            fret = pitch - OPEN_BY_SOURCE[source]
            if not 0 <= fret <= 24:
                continue  # pitch fuera de la cuerda (error de anotación)
            start = to_ticks(note["time"])
            dur = max(1, to_ticks(note["time"] + note["duration"]) - start)
            rows.append((start, pitch, dur, string, fret))
    rows.sort(key=lambda r: (r[0], r[1]))
    if len(rows) < min_notes:
        return None
    name = os.path.basename(path)[: -len(".jams")]
    return {
        "notes": [(pitch, dur) for _, pitch, dur, _, _ in rows],
        "tabs": [(Position(string, fret), dur) for _, _, dur, string, fret in rows],
        "starts": [start for start, _, _, _, _ in rows],
        "meta": {"song": name, **parse_name(name), "n_notes": len(rows)},
    }


def load_guitarset(root: str, min_notes: int = 8) -> List[dict]:
    """Todos los fragmentos de GuitarSet (orden alfabético)."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*.jams"))):
        song = load_jams(path, min_notes)
        if song is not None:
            out.append(song)
    return out


def progression(song: dict) -> str:
    """Progresión de acordes de un fragmento: `00_BN1-129-Eb_comp` -> `BN1`.
    Los 6 guitarristas tocan las mismas 15 progresiones (cada una a 2 tempos y
    tonalidades) → es la unidad que no puede repartirse entre train y test."""
    return song["meta"]["song"].split("_")[1].split("-")[0]


def split_guitarset(songs, seed: int = 42):
    """Train/test por progresión, estratificado por estilo: de las 3
    progresiones de cada estilo, 2 a train y 1 a test (elegida con semilla).
    Test = música nunca vista (tocada por los mismos 6 guitarristas)."""
    import random
    rng = random.Random(seed)
    test_progs = set()
    for style in sorted(STYLES):
        progs = sorted({progression(s) for s in songs if s["meta"]["style"] == style})
        test_progs.add(rng.choice(progs))
    return ([s for s in songs if progression(s) not in test_progs],
            [s for s in songs if progression(s) in test_progs])


def snap_durations(songs, durations):
    """Ajusta cada duración a la más cercana de `durations` (vocabulario de
    DadaGP): GuitarSet es interpretación real (duraciones no cuantizadas)."""
    ds = sorted(set(durations))
    snap = lambda d: min(ds, key=lambda x: abs(x - d))
    return [{**s, "notes": [(p, snap(d)) for p, d in s["notes"]],
             "tabs": [(t, snap(d)) for t, d in s["tabs"]]} for s in songs]
