"""Tests del reparto del dataset mezclado (sin fugas y estratificado)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from build_mixed_dataset import split_groups, split_guitarset_by_style  # noqa: E402
from src.data.guitarset import STYLES, parse_name


def _gs():
    return [{"meta": {"song": f"0{p}_{st}{k}-100-C_{part}", **parse_name(f"0{p}_{st}{k}-100-C_{part}")}}
            for st in STYLES for k in (1, 2, 3) for p in range(6) for part in ("comp", "solo")]


def test_guitarset_estratificado_por_estilo():
    songs = _gs()
    a = split_guitarset_by_style(songs, seed=0)
    assert sorted(a.values()).count("test") == 2 and sorted(a.values()).count("val") == 2
    held = [p for p, sp in a.items() if sp != "train"]
    assert len({p.rstrip("123") for p in held}) == 4          # 4 estilos distintos
    for st in STYLES:                                          # cada estilo ≥ 2 en train
        assert sum(1 for p, sp in a.items() if p.startswith(st) and sp == "train") >= 2


def test_split_groups_sin_solape_y_proporciones():
    a = split_groups([f"obra{i}" for i in range(100)], seed=0)
    assert list(a.values()).count("test") == 10 and list(a.values()).count("val") == 10
    assert split_groups([f"obra{i}" for i in range(100)], seed=0) == a
    assert set(split_groups(["a", "b", "c"]).values()) >= {"val", "test"}   # al menos 1 grupo
