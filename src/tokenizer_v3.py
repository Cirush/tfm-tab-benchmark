"""Tokenización según el encoding v3 del Fretting-Transformer (arXiv:2506.14223).

Formato por nota:
  Entrada : NOTE_ON<pitch> TIME_SHIFT<dur> NOTE_OFF<pitch>
  Salida  : TAB<string,fret> TIME_SHIFT<dur>

Las duraciones se expresan en ticks (960 por negra en DadaGP). El vocabulario de
TIME_SHIFT se construye a partir de las duraciones vistas en train
(`from_data`, como el tokenizador word-level del paper); las duraciones no
vistas se aproximan a la más cercana.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple
from .fretboard import Position

# Duraciones cuantizadas por defecto (ticks, 960 = negra):
# fusa, semicorchea, corchea, corchea con puntillo, negra, negra puntillo, blanca, redonda
DEFAULT_DURATIONS = (120, 240, 480, 720, 960, 1440, 1920, 3840)

PITCH_MIN, PITCH_MAX = 30, 100  # rango holgado (7 cuerdas / afinaciones bajas)
STRINGS = range(1, 8)           # DadaGP soporta hasta 7 cuerdas
FRETS = range(0, 25)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


@dataclass
class V3Tokenizer:
    durations: Tuple[int, ...] = DEFAULT_DURATIONS
    vocab: Dict[str, int] = field(default_factory=dict)
    inv_vocab: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        tokens = [PAD, BOS, EOS, UNK]
        tokens += [f"NOTE_ON<{p}>" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        tokens += [f"NOTE_OFF<{p}>" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        tokens += [f"TIME_SHIFT<{d}>" for d in self.durations]
        tokens += [f"TAB<{s},{f}>" for s in STRINGS for f in FRETS]
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    # ---- cuantización ----
    def quantize(self, ticks: int) -> int:
        """Duración más cercana del conjunto cerrado."""
        return min(self.durations, key=lambda d: abs(d - ticks))

    # ---- texto ----
    def encode_input(self, notes: Sequence[Tuple[int, int]]) -> List[str]:
        """notes: [(pitch, dur_ticks)] -> tokens de entrada v3."""
        out = []
        for pitch, dur in notes:
            d = self.quantize(dur)
            out += [f"NOTE_ON<{pitch}>", f"TIME_SHIFT<{d}>", f"NOTE_OFF<{pitch}>"]
        return out

    def encode_target(self, tabs: Sequence[Tuple[Position, int]]) -> List[str]:
        """tabs: [(Position, dur_ticks)] -> tokens objetivo v3."""
        out = []
        for pos, dur in tabs:
            d = self.quantize(dur)
            out += [f"TAB<{pos.string},{pos.fret}>", f"TIME_SHIFT<{d}>"]
        return out

    def decode_target(self, tokens: Sequence[str]) -> List[Position]:
        """Extrae las posiciones de una secuencia objetivo (ignora time shifts
        y tokens especiales; tolera salida ruidosa del modelo)."""
        out = []
        for t in tokens:
            if t.startswith("TAB<") and t.endswith(">"):
                try:
                    s, f = t[4:-1].split(",")
                    out.append(Position(int(s), int(f)))
                except ValueError:
                    continue  # token malformado generado por el modelo
        return out

    # ---- ids ----
    def to_ids(self, tokens: Sequence[str], add_special: bool = True) -> List[int]:
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
        if add_special:
            ids = [self.vocab[BOS]] + ids + [self.vocab[EOS]]
        return ids

    def to_tokens(self, ids: Sequence[int]) -> List[str]:
        specials = {self.vocab[PAD], self.vocab[BOS], self.vocab[EOS]}
        return [self.inv_vocab[i] for i in ids if i in self.inv_vocab and i not in specials]

    # ---- vocabulario de duraciones desde los datos (como el paper) ----
    @classmethod
    def from_data(cls, durations: Iterable[int], min_count: int = 1) -> "V3Tokenizer":
        """Tokenizador word-level como el paper: un TIME_SHIFT por cada duración
        vista en train (≥ min_count veces). Las no vistas se mapean a la más
        cercana con `quantize`."""
        counts = Counter(durations)
        return cls(durations=tuple(sorted(d for d, c in counts.items() if c >= min_count and d > 0)))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"durations": list(self.durations)}, fh)

    @classmethod
    def load(cls, path: str) -> "V3Tokenizer":
        with open(path, encoding="utf-8") as fh:
            return cls(durations=tuple(json.load(fh)["durations"]))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]
