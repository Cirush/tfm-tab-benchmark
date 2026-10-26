"""Entrenamiento de los modelos transformer: T5 (por defecto), GPT2 (`--arch gpt2`)
o BERT (`--arch bert`).

Fiel al paper donde lo especifica (config T5 reducida, Adafactor con learning
rate auto-adaptativo, secuencias de 512 tokens, inferencia por trozos de 20
notas + post-procesado). Lo no especificado (batch, épocas, early stopping) se
elige aquí y se registra en `history.json`.

Por época: loss de train, loss de val (teacher forcing sobre ventanas de val);
se guarda el mejor modelo por val loss y se para si no mejora en `--patience`
épocas. Al final: evaluación con generación real (trozos + post-procesado) sobre
`--eval-tracks` pistas de val, con las 3 etapas del post-procesado.

Entrada: <data>/{train,val}.jsonl.gz (p. ej. data/mixed, de
scripts/build_mixed_dataset.py).
Uso:
    python3.12 scripts/train.py --data data/mixed --durations-from-data --limit-train 200 --epochs 2
    python3.12 scripts/train.py --data data/mixed --durations-from-data --out runs/mixed_t5
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.optimization import Adafactor

from src.data.chunking import max_notes_for, random_windows, training_windows
from src.data.dadagp_gp import load_tracks
from src.models import bert_fretting, gpt2_fretting, t5_fretting
from src.tokenizer_v3 import V3Tokenizer

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def mean_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for src, mask, labels in loader:
        out = model(input_ids=src.to(device), attention_mask=mask.to(device),
                    labels=labels.to(device))
        k = (labels != -100).sum().item()  # media por token, no por lote
        total += out.loss.item() * k
        n += k
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "data", "mixed"))
    ap.add_argument("--out", default=os.path.join(PROJECT, "runs", "t5"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-train", type=int, default=None, help="nº de pistas de train (prototipo)")
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--eval-tracks", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-notes", type=int, default=max_notes_for(),
                    help="notas por ventana de entrenamiento (defecto: 170 = 512 tokens)")
    ap.add_argument("--arch", default="t5", choices=["t5", "gpt2", "bert"],
                    help="t5 = traducción; gpt2 = completar texto; bert = rellenar huecos (§4.4 del paper)")
    ap.add_argument("--model-config", default="{}",
                    help='JSON con cambios a la configuración del modelo, p. ej. '
                         '\'{"hidden_size": 128, "num_hidden_layers": 8}\' (ablación de tamaño)')
    ap.add_argument("--durations-from-data", action="store_true",
                    help="vocabulario de TIME_SHIFT desde las duraciones de train (réplica del paper)")
    ap.add_argument("--window-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None,
                    help="ventanas de longitud aleatoria en [MIN, MAX] notas (ignora --max-notes)")
    ap.add_argument("--warmup-init", action=argparse.BooleanOptionalAction, default=False,
                    help="Adafactor: arrancar el lr en 1e-6 y subir linealmente (lento)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    train_tracks = load_tracks(os.path.join(args.data, "train.jsonl.gz"))[: args.limit_train]
    val_tracks = load_tracks(os.path.join(args.data, "val.jsonl.gz"))[: args.limit_val]
    if args.durations_from_data:
        # tokenizador word-level como el paper: un TIME_SHIFT por duración vista en train
        tok = V3Tokenizer.from_data(d for s in train_tracks for _, d in s["notes"])
    else:
        tok = V3Tokenizer()
    tok.save(os.path.join(args.out, "tokenizer.json"))  # la inferencia debe usar el mismo vocabulario
    max_notes = args.max_notes
    if args.window_range:
        # longitudes aleatorias (fijas para todo el entrenamiento, semilla fija):
        # un mismo modelo sirve para distintos protocolos de inferencia
        rng = random.Random(args.seed)
        lo, hi = args.window_range
        make = lambda s: random_windows(s, lo, hi, rng)
    else:
        make = lambda s: training_windows(s, max_notes)
    arch = {"t5": t5_fretting, "gpt2": gpt2_fretting, "bert": bert_fretting}[args.arch]
    WindowDataset = arch.WindowDataset
    train_ds = WindowDataset([w for s in train_tracks for w in make(s)], tok)
    val_ds = WindowDataset([w for s in val_tracks for w in make(s)], tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=val_ds.collate)

    model = arch.build_model(tok, **json.loads(args.model_config)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Adafactor con learning rate auto-adaptativo, como el paper
    opt = Adafactor(model.parameters(), lr=None, relative_step=True,
                    scale_parameter=True, warmup_init=args.warmup_init)

    history = {
        "config": {**vars(args), "model": model.config.to_diff_dict(), "params": n_params,
                   "device": str(device), "max_notes": max_notes, "vocab_size": tok.vocab_size,
                   "train_tracks": len(train_tracks), "train_windows": len(train_ds),
                   "val_tracks": len(val_tracks), "val_windows": len(val_ds)},
        "epochs": [],
    }
    print(f"device={device} | params={n_params / 1e6:.2f}M | "
          f"train {len(train_tracks)} pistas / {len(train_ds)} ventanas | "
          f"val {len(val_tracks)} pistas / {len(val_ds)} ventanas")

    best, bad_epochs = float("inf"), 0
    best_dir = os.path.join(args.out, "best")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for src, mask, labels in train_dl:
            out = model(input_ids=src.to(device), attention_mask=mask.to(device),
                        labels=labels.to(device))
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += out.loss.item()
            n += 1
        train_loss = total / n
        val_loss = mean_loss(model, val_dl, device)
        improved = val_loss < best
        if improved:
            best, bad_epochs = val_loss, 0
            model.save_pretrained(best_dir)
        else:
            bad_epochs += 1
        history["epochs"].append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                                  "seconds": time.time() - t0})
        with open(os.path.join(args.out, "history.json"), "w") as fh:
            json.dump(history, fh, indent=1)
        print(f"epoch {ep:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"{time.time() - t0:.0f}s{'  *' if improved else ''}", flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stopping: {args.patience} épocas sin mejorar val loss")
            break

    # Evaluación con generación real sobre pistas de val completas
    model = arch.load_model(best_dir).to(device)
    t0 = time.time()
    res = arch.evaluate_tracks(model, tok, val_tracks[: args.eval_tracks])
    res["seconds"] = time.time() - t0
    history["eval_val"] = res
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=1)
    print(f"\nEvaluación val ({res['n_tracks']} pistas, {res['n_notes']} notas, {res['seconds']:.0f}s):")
    for stage in ("raw", "window", "final"):
        print(f"  {stage:6s}  tab {res[f'tab_acc_{stage}']:.2%}  pitch {res[f'pitch_acc_{stage}']:.2%}")
    print(f"  difficulty pred {res['difficulty_pred']:.2f} | GT {res['difficulty_gt']:.2f}")


if __name__ == "__main__":
    main()
