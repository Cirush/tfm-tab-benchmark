"""Generador de datos sintéticos para las pruebas automáticas del pipeline.

Genera melodías con movimiento melódico plausible y usa A* como oráculo de
ground truth (digitación biomecánicamente razonable). NO sustituye a DadaGP:
sirve exclusivamente para verificar el pipeline de entrenamiento de punta a
punta antes de tener los datos reales.

Un modelo entrenado con estas digitaciones solo aprendería a imitar a A*, así que
estos datos nunca se usan para medir resultados.
"""

import random
from typing import List, Tuple
from ..fretboard import Fretboard
from ..models.astar import AStarFretting
from ..tokenizer_v3 import DEFAULT_DURATIONS


def synth_melody(rng: random.Random, min_len=8, max_len=32) -> List[Tuple[int, int]]:
    """Melodía sintética: [(pitch, dur_ticks)] con pasos melódicos realistas."""
    n = rng.randint(min_len, max_len)
    pitch = rng.randint(50, 76)
    steps = [-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7]
    notes = []
    for _ in range(n):
        pitch = max(40, min(84, pitch + rng.choice(steps)))
        dur = rng.choice(DEFAULT_DURATIONS[:6])  # duraciones comunes
        notes.append((pitch, dur))
    return notes


def make_dataset(n_songs: int, seed: int = 0):
    """Genera pares (entrada, ground truth) usando A* como oráculo.

    Devuelve lista de dicts: {"notes": [(pitch,dur)], "tabs": [(Position,dur)]}
    """
    rng = random.Random(seed)
    fb = Fretboard()
    oracle = AStarFretting(fb)
    data = []
    for _ in range(n_songs):
        notes = synth_melody(rng)
        positions = oracle.predict([p for p, _ in notes])
        tabs = [(pos, dur) for pos, (_, dur) in zip(positions, notes)]
        data.append({"notes": notes, "tabs": tabs})
    return data


def train_val_test_split(data, val_frac=0.1, test_frac=0.1, seed=0):
    """Split por 'canción' completa — nunca por fragmento (evita leakage)."""
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_test = int(len(data) * test_frac)
    n_val = int(len(data) * val_frac)
    test = [data[i] for i in idx[:n_test]]
    val = [data[i] for i in idx[n_test:n_test + n_val]]
    train = [data[i] for i in idx[n_test + n_val:]]
    return train, val, test
