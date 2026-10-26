"""Tab accuracy por estilo y por parte (acompañamiento/solo) en el test de GuitarSet.

Uso: python3.12 scripts/eval_por_estilo.py --data data/mixed --split test_guitarset \
        --run runs/mixed_t5 --run ... --name definitivo_guitarset_por_estilo
Salida: experiments/<name>.json  (acierto ponderado por nota en cada grupo)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dadagp_gp import load_tracks
from src.fretboard import Fretboard
from src.models.astar_burlet import AStarBurlet
from src.models.baseline_minfret import MinFretBaseline

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def by_group(preds, songs):
    hits, total = defaultdict(int), defaultdict(int)
    for pred, s in zip(preds, songs):
        m = s["meta"]
        for key in (m["style"], m["part"], f"{m['style']} {m['part']}", "total"):
            for p, (g, _) in zip(pred, s["tabs"]):
                total[key] += 1
                hits[key] += int(p == g)
    return {k: {"tab_acc": hits[k] / total[k], "n_notes": total[k]} for k in total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "data", "mixed"))
    ap.add_argument("--split", default="test_guitarset")
    ap.add_argument("--run", dest="runs", action="append", default=[])
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    songs = load_tracks(os.path.join(args.data, f"{args.split}.jsonl.gz"))
    fb = Fretboard()
    pitches = [[p for p, _ in s["notes"]] for s in songs]
    res = {"baseline": by_group([MinFretBaseline().predict(p) for p in pitches], songs)}
    astar = AStarBurlet(fb)
    res["A*"] = by_group([astar.predict(p, s.get("starts")) for p, s in zip(pitches, songs)], songs)

    from src.models import bert_fretting, gpt2_fretting, t5_fretting
    from src.tokenizer_v3 import V3Tokenizer
    from train import pick_device
    for run in args.runs:
        with open(os.path.join(run, "history.json")) as fh:
            arch_name = json.load(fh)["config"].get("arch", "t5")
        arch = {"t5": t5_fretting, "gpt2": gpt2_fretting, "bert": bert_fretting}[arch_name]
        tok = V3Tokenizer.load(os.path.join(run, "tokenizer.json"))
        model = arch.load_model(os.path.join(run, "best")).to(pick_device("auto"))
        out = arch.predict_tracks(model, tok, [s["notes"] for s in songs], fb=fb)
        res[os.path.basename(os.path.normpath(run))] = by_group([o["final"] for o in out], songs)

    path = os.path.join(PROJECT, "experiments", f"{args.name}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    groups = sorted(next(iter(res.values())), key=lambda k: (k == "total", k))
    print(f"{'grupo':16s}" + "".join(f"{m[:14]:>15s}" for m in res))
    for g in groups:
        n = res["baseline"][g]["n_notes"]
        print(f"{g:16s}" + "".join(f"{res[m][g]['tab_acc']:15.1%}" for m in res) + f"   ({n} notas)")
    print("Guardado en", path)


if __name__ == "__main__":
    main()
