"""Loader de DadaGP desde los ficheros GuitarPro originales (réplica del paper).

El Fretting-Transformer (arXiv:2506.14223) filtra DadaGP a las pistas de
guitarra acústica usando el instrumento MIDI: *"25 for nylon stringed guitars
and 26 for steel stringed guitars"* (numeración General MIDI 1-indexada). Los
`.tokens.txt` de DadaGP agrupan las guitarras en clean*/distorted* y no
distinguen acústicas, así que aquí se leen los `.gp3/.gp4/.gp5` originales con
PyGuitarPro, que sí exponen el instrumento de cada pista.

Salida: mismo formato que `dadagp.py` -> `{"notes": [(pitch, dur)],
"tabs": [(Position, dur)], "meta": {...}}`, así el resto del pipeline no cambia.

Decisiones de diseño:
- Unidad de ejemplo = PISTA (el paper cuenta pistas: 2.301 de 47.039). Una
  canción puede aportar varias pistas acústicas.
- Solo afinación estándar, 6 cuerdas y sin cejilla: el tokenizador v3 no
  codifica afinación ni cejilla, así que con otras afinaciones el mismo pitch
  correspondería a trastes distintos sin que el modelo pueda saberlo. El paper
  trata afinación/cejilla en un experimento aparte ("conditioned").
- Se extraen todas las notas de la pista (`track_notes_poly`), juntando todas
  sus voces, como en la conversión a MIDI del paper.
- Notas muertas (x) se descartan: no tienen altura.
- Se lee la partitura tal como está escrita (sin desplegar repeticiones).
- DadaGP contiene ficheros duplicados exactos en carpetas distintas. Se detectan
  por hash del contenido (`dedup`) y, como hace el paper, se conserva un único
  fichero por título normalizado (`dedup_paper`, `song_key`), de modo que dos
  versiones de una misma canción no puedan acabar una en train y otra en test.
- Split 80/10/10 por fichero, estratificado por tonalidad (`split_paper`).
"""

import gzip
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import guitarpro

from ..fretboard import Position, STANDARD_TUNING
from ..tokenizer_v3 import PITCH_MIN, PITCH_MAX

# Programas MIDI 0-indexados (PyGuitarPro) = GM 25 y 26 en 1-indexado (paper).
ACOUSTIC_PROGRAMS = {24: "nylon", 25: "steel"}
# Todas las guitarras GM (25-32 en 1-indexado): denominador comparable al del paper.
GUITAR_PROGRAMS = range(24, 32)

TOKENS_SUFFIX = ".tokens.txt"

# Paper: "additional keyword-based filtering is applied using guitar-related
# terms in various languages". Lista no publicada -> "guitarra" en los idiomas
# frecuentes de DadaGP y abreviaturas (calibrado: nombres como "Track 1",
# "Piste 1" o "Acoustic" quedan fuera).
GUITAR_KEYWORDS = re.compile(
    r"guit|gtr|gitar|chitarr|viol[aã]o|kitar|gitaar|gitár|ギター|吉他|гитар", re.I
)


@dataclass
class TrackInfo:
    """Metadatos de una pista acústica encontrada al escanear el dataset."""
    song: str            # ruta relativa del .gp original (id de canción)
    track: int           # número de pista en el fichero GP (1-indexado)
    program: int         # instrumento MIDI 0-indexado
    tuning: Tuple[int, ...]
    capo: int
    artist: str
    md5: str = ""        # hash del fichero GP (detección de duplicados)
    name: str = ""       # nombre de la pista en el GP (filtro por palabra clave del paper)
    key: str = ""        # armadura del primer compás (estratificación del split del paper)

    @property
    def usable(self) -> bool:
        """Afinación estándar de 6 cuerdas y sin cejilla."""
        return tuple(self.tuning) == STANDARD_TUNING and self.capo == 0


# --------------------------------------------------------------------------- #
# Índice de canciones y escaneo
# --------------------------------------------------------------------------- #
def list_songs(root: str) -> List[Tuple[str, str]]:
    """[(ruta relativa del .gp original, artista)] de todo DadaGP (train + val).

    El original es el `tokens.txt` sin el sufijo (p. ej. `X.gp4.tokens.txt` ->
    `X.gp4`); los `.pygp.gp5` y `.gp2tokens2gp.gp5` son regeneraciones de DadaGP.
    """
    with open(os.path.join(root, "_DadaGP_all_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    out = []
    for tokens_rel, m in meta.items():
        if tokens_rel.endswith(TOKENS_SUFFIX):
            artist = m.get("artist_token", "artist:unknown_artist").split(":", 1)[1]
            out.append((tokens_rel[: -len(TOKENS_SUFFIX)], artist))
    return sorted(out)


def scan_song(root: str, song_rel: str, artist: str = "") -> Tuple[int, int, List[TrackInfo]]:
    """Abre un .gp y devuelve (nº de pistas, nº de pistas de guitarra de
    cualquier tipo, pistas acústicas).

    Lanza la excepción de PyGuitarPro si el fichero no se puede leer.
    """
    path = os.path.join(root, song_rel)
    with open(path, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    song = guitarpro.parse(path)
    key = song.measureHeaders[0].keySignature.name if song.measureHeaders else ""
    acoustic = []
    for t in song.tracks:
        if t.isPercussionTrack or t.channel.instrument not in ACOUSTIC_PROGRAMS:
            continue
        acoustic.append(TrackInfo(
            song=song_rel,
            track=t.number,
            program=t.channel.instrument,
            tuning=tuple(s.value for s in t.strings),
            capo=t.offset,
            artist=artist,
            md5=md5,
            name=t.name or "",
            key=key,
        ))
    n_guitar = sum(
        1 for t in song.tracks
        if not t.isPercussionTrack and t.channel.instrument in GUITAR_PROGRAMS
    )
    return len(song.tracks), n_guitar, acoustic


def dedup(infos: Sequence[TrackInfo]) -> List[TrackInfo]:
    """Elimina canciones duplicadas exactas (mismo md5): se queda con la primera
    ruta en orden alfabético y descarta todas las pistas de las copias."""
    keep: Dict[str, str] = {}
    for song, md5 in sorted({(i.song, i.md5) for i in infos}):
        keep.setdefault(md5, song)
    kept_songs = set(keep.values())
    return [i for i in infos if i.song in kept_songs]


def song_key(song_rel: str) -> str:
    """Título normalizado de una canción (agrupa versiones de la misma canción).

    `Artista/Artista - Título (2).gp4` -> `artista titulo`: minúsculas, sin
    extensión, sin marcadores de versión ("(2)", "ver 2", "v2") ni números
    finales. Títulos genéricos iguales de canciones distintas acaban agrupados:
    es inocuo (solo hace el grupo más grande), lo contrario sería leakage.
    """
    name = os.path.basename(song_rel).lower()
    name = re.sub(r"\.gp[345]$", "", name)
    name = re.sub(r"\(\d+\)|\bver\.? ?\d+\b|\bv\d+\b|\d+$", "", name)
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def save_index(
    path: str, infos: Sequence[TrackInfo], stats: Dict,
    splits: Optional[Dict[str, List[TrackInfo]]] = None,
) -> None:
    """Guarda todas las pistas acústicas, estadísticas y (opcional) el split."""
    data = {"stats": stats, "tracks": [asdict(i) for i in infos]}
    if splits is not None:
        data["splits"] = {k: [asdict(i) for i in v] for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)


def _info(d: dict) -> TrackInfo:
    return TrackInfo(**{**d, "tuning": tuple(d["tuning"])})


def load_index(path: str) -> Tuple[List[TrackInfo], Dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [_info(d) for d in data["tracks"]], data["stats"]


# --------------------------------------------------------------------------- #
# Extracción de notas
# --------------------------------------------------------------------------- #
def save_tracks(path: str, songs: Sequence[dict]) -> None:
    """Guarda pistas extraídas en JSONL comprimido (1 pista por línea).

    Formato compacto: `notes` = [[pitch, dur]], `pos` = [[cuerda, traste]] y,
    si existe, `starts` = [inicio en ticks] (lo necesita el A* con acordes).
    """
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for s in songs:
            row = {
                "meta": s["meta"],
                "notes": [[p, d] for p, d in s["notes"]],
                "pos": [[pos.string, pos.fret] for pos, _ in s["tabs"]],
            }
            if "starts" in s:
                row["starts"] = list(s["starts"])
            fh.write(json.dumps(row) + "\n")


def load_tracks(path: str) -> List[dict]:
    """Inverso de `save_tracks` -> [{"notes","tabs","meta"}]."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            notes = [tuple(n) for n in d["notes"]]
            tabs = [(Position(s, f), dur) for (s, f), (_, dur) in zip(d["pos"], notes)]
            song = {"notes": notes, "tabs": tabs, "meta": d["meta"]}
            if "starts" in d:
                song["starts"] = d["starts"]
            out.append(song)
    return out


# --------------------------------------------------------------------------- #
# Réplica del paper: filtro por nombre, notas polifónicas, dedup y split
# --------------------------------------------------------------------------- #
def is_guitar_named(info: TrackInfo) -> bool:
    return bool(GUITAR_KEYWORDS.search(info.name or ""))


def track_notes_poly(track) -> List[Tuple[int, int, int, int, int]]:
    """TODAS las notas de una pista, como en la conversión GP->MIDI del paper:
    [(start, end, pitch, cuerda, traste)] ordenadas por (start, pitch).

    - Una nota ligada (tie) no es un ataque nuevo: alarga la nota abierta en esa
      misma cuerda (en MIDI es una sola nota más larga).
    - Se descartan notas muertas (x) y silencios. Las notas de adorno (grace) no
      se añaden (el paper no las menciona).
    - Ticks fraccionarios de tresillos se redondean al tick entero.
    """
    open_by_string: Dict[int, list] = {}
    notes: List[list] = []
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                start = round(beat.start)
                end = start + round(beat.duration.time)
                for n in beat.notes:
                    if n.type == guitarpro.NoteType.tie:
                        prev = open_by_string.get(n.string)
                        if prev is not None:
                            prev[1] = max(prev[1], end)
                        continue
                    if n.type != guitarpro.NoteType.normal:
                        continue
                    rec = [start, end, n.realValue, n.string, n.value]
                    notes.append(rec)
                    open_by_string[n.string] = rec
    notes.sort(key=lambda r: (r[0], r[2]))
    return [tuple(r) for r in notes]


def load_track_poly(root: str, info: TrackInfo, min_notes: int = 8) -> Optional[dict]:
    """Pista con todas sus notas -> {"notes": [(pitch, dur)], "tabs": [(Position,
    dur)], "meta"} con dur = end - start (TIME_SHIFT del paper = duración)."""
    song = guitarpro.parse(os.path.join(root, info.song))
    track = next(t for t in song.tracks if t.number == info.track)
    rows = [r for r in track_notes_poly(track)
            if 1 <= r[3] <= 6 and 0 <= r[4] <= 24 and PITCH_MIN <= r[2] <= PITCH_MAX]
    if len(rows) < min_notes:
        return None
    return {
        "notes": [(pitch, end - start) for start, end, pitch, _, _ in rows],
        "tabs": [(Position(st, fr), end - start) for start, end, _, st, fr in rows],
        "starts": [start for start, _, _, _, _ in rows],
        "meta": {**asdict(info), "n_notes": len(rows)},
    }


def dedup_paper(infos: Sequence[TrackInfo]) -> List[TrackInfo]:
    """Paper: "Duplicate tracks are removed ... by matching metadata or file
    names" -> por cada título normalizado (`song_key`) se conserva un único
    fichero (el primero en orden alfabético) con todas sus pistas."""
    keep: Dict[str, str] = {}
    for song in sorted({i.song for i in infos}):
        keep.setdefault(song_key(song), song)
    kept = set(keep.values())
    return [i for i in infos if i.song in kept]


def split_paper(
    infos: Sequence[TrackInfo],
    fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[TrackInfo]]:
    """Split 80/10/10 "maintaining diversity in key" (paper): estratificado por
    armadura. La unidad es el fichero (todas sus pistas al mismo conjunto) para
    no repartir partes de una misma canción entre train y test."""
    by_key: Dict[str, List[str]] = {}
    for song, key in sorted({(i.song, i.key) for i in infos}):
        by_key.setdefault(key, []).append(song)
    rng = random.Random(seed)
    assign: Dict[str, str] = {}
    for key in sorted(by_key):
        songs = by_key[key]
        rng.shuffle(songs)
        n_train = round(len(songs) * fracs[0])
        n_val = round(len(songs) * fracs[1])
        for k, song in enumerate(songs):
            assign[song] = "train" if k < n_train else "val" if k < n_train + n_val else "test"
    out: Dict[str, List[TrackInfo]] = {"train": [], "val": [], "test": []}
    for i in infos:
        out[assign[i.song]].append(i)
    return out
