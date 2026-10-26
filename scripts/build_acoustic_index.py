"""Escanea DadaGP y construye el índice de pistas de guitarra acústica.

Replica el filtro del Fretting-Transformer (instrumentos GM 25 nylon / 26 acero)
leyendo los .gp originales. Guarda en `data/acoustic_index.json` (no versionado:
se regenera con este script) las pistas acústicas encontradas y estadísticas
para comparar con las del paper (47.039 pistas totales -> 2.301 acústicas).

Uso:
    python3.12 scripts/build_acoustic_index.py            # todo el dataset
    python3.12 scripts/build_acoustic_index.py --limit 300
"""

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import dedup, list_songs, save_index, scan_song, song_key
from src.fretboard import STANDARD_TUNING

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(PROJECT), "DadaGP-v1.1")
OUT = os.path.join(PROJECT, "data", "acoustic_index.json")


def _scan(args):
    song_rel, artist = args
    try:
        return (*scan_song(ROOT, song_rel, artist), None)
    except Exception as e:  # ficheros GP corruptos o versiones no soportadas
        return 0, 0, [], f"{song_rel}: {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    songs = list_songs(ROOT)
    if args.limit:
        songs = songs[: args.limit]
    print(f"Canciones a escanear: {len(songs)} | workers: {args.workers}")

    t0 = time.time()
    total_tracks, guitar_tracks, infos, errors = 0, 0, [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (n, ng, found, err) in enumerate(ex.map(_scan, songs, chunksize=32), 1):
            total_tracks += n
            guitar_tracks += ng
            infos.extend(found)
            if err:
                errors.append(err)
            if k % 2000 == 0:
                print(f"  {k}/{len(songs)}  ({time.time() - t0:.0f}s)")

    usable_all = [i for i in infos if i.usable]
    usable = dedup(usable_all)
    stats = {
        "songs_scanned": len(songs),
        "songs_unreadable": len(errors),
        "tracks_total": total_tracks,
        "guitar_tracks": guitar_tracks,  # comparable a las 47.039 del paper
        "acoustic_tracks": len(infos),
        "acoustic_songs": len({i.song for i in infos}),
        "by_program": dict(Counter(i.program for i in infos)),
        "standard_tuning": sum(tuple(i.tuning) == STANDARD_TUNING for i in infos),
        "with_capo": sum(i.capo != 0 for i in infos),
        "not_6_strings": sum(len(i.tuning) != 6 for i in infos),
        "duplicate_songs_removed": len({i.song for i in usable_all}) - len({i.song for i in usable}),
        "usable_tracks": len(usable),
        "usable_songs": len({i.song for i in usable}),
        "usable_titles": len({song_key(i.song) for i in usable}),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_index(OUT, infos, stats)

    print(f"\nTiempo: {time.time() - t0:.0f}s")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if errors:
        print(f"\nPrimeros errores de lectura: {errors[:5]}")
    print(f"\nÍndice guardado en {OUT}")


if __name__ == "__main__":
    main()
