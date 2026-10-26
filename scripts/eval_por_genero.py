"""Tab accuracy por macrogénero en el test de DadaGP.

El género de cada canción sale de los metadatos de DadaGP
(`_DadaGP_all_metadata.json`, etiquetas del artista) y se agrupa en
macrogéneros con `src/data/generos.py`.

Uso: python3.12 scripts/eval_por_genero.py --data data/mixed \
        --run runs/mixed_t5 --run ... --name definitivo_dadagp_por_genero
     python3.12 scripts/eval_por_genero.py --reagrupar --name definitivo_dadagp_por_genero
     (rehace la agrupación con los aciertos guardados, sin volver a predecir)
Salida: experiments/<name>.json  (acierto ponderado por nota, pistas y artistas
por grupo, proporción de notas a partir del traste 5 y notas por ataque en la
tablatura real, y aciertos por pista para poder reagrupar sin volver a predecir)

Como en varios grupos uno o dos artistas aportan buena parte de las notas, cada
acierto lleva un intervalo de confianza del 95 % por bootstrap remuestreando
artistas (no pistas ni notas, que no son independientes entre sí). También se
da la diferencia emparejada de cada método con A* (mismas pistas), con
su intervalo: si no contiene el 0, la diferencia es significativa.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dadagp_gp import load_tracks
from src.data.generos import macro_cancion
from src.fretboard import Fretboard
from src.models.astar_burlet import AStarBurlet
from src.models.baseline_minfret import MinFretBaseline

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = "A*"  # referencia de las diferencias emparejadas


def artista(cancion):
    """Carpeta del artista en DadaGP (el campo de metadatos agrupa muchos como "unknown")."""
    return cancion.split("/")[1]


def hits_por_pista(preds, songs):
    return [sum(int(p == g) for p, (g, _) in zip(pred, s["tabs"])) for pred, s in zip(preds, songs)]


def intervalos(pistas, idx, metodos, ref, n_boot=2000, seed=0):
    """IC del 95 % (bootstrap remuestreando artistas con reemplazo) de la tab
    accuracy de cada método y de su diferencia emparejada con `ref`."""
    por_artista = defaultdict(lambda: [0, defaultdict(int)])
    for i in idx:
        a = por_artista[pistas[i]["artista"]]
        a[0] += pistas[i]["n_notas"]
        for m in metodos:
            a[1][m] += pistas[i]["aciertos"][m]
    artistas = list(por_artista.values())
    rng = random.Random(seed)
    acc, dif = defaultdict(list), defaultdict(list)
    for _ in range(n_boot):
        sel = rng.choices(artistas, k=len(artistas))
        n = sum(a[0] for a in sel)
        for m in metodos:
            acc[m].append(sum(a[1][m] for a in sel) / n)
            dif[m].append(sum(a[1][m] - a[1][ref] for a in sel) / n)

    def ic(v):
        v = sorted(v)
        return v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]
    return {m: ic(acc[m]) for m in metodos}, {m: ic(dif[m]) for m in metodos if m != ref}


def agrupar(pistas, metodos):
    grupos = defaultdict(list)
    for i, p in enumerate(pistas):
        grupos[p["genero"]].append(i)
        grupos["total"].append(i)
    out = {}
    for g, idx in grupos.items():
        n = sum(pistas[i]["n_notas"] for i in idx)
        out[g] = {"n_pistas": len(idx), "n_artistas": len({pistas[i]["artista"] for i in idx}),
                  "n_notas": n,
                  "frac_traste_5": sum(pistas[i]["n_traste_5"] for i in idx) / n,
                  "notas_por_ataque": n / sum(pistas[i]["n_ataques"] for i in idx),
                  "tab_acc": {m: sum(pistas[i]["aciertos"][m] for i in idx) / n for m in metodos}}
        acc = out[g]["tab_acc"]
        out[g]["ic95"], ic_dif = intervalos(pistas, idx, metodos, REF)
        out[g][f"dif_vs_{REF}"] = {m: {"dif": acc[m] - acc[REF], "ic95": ic_dif[m]}
                                   for m in metodos if m != REF}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "data", "mixed"))
    ap.add_argument("--split", default="test_dadagp")
    ap.add_argument("--dadagp", default=os.path.join(PROJECT, "..", "DadaGP-v1.1"))
    ap.add_argument("--run", dest="runs", action="append", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--reagrupar", action="store_true")
    args = ap.parse_args()
    path = os.path.join(PROJECT, "experiments", f"{args.name}.json")

    if args.reagrupar:
        with open(path) as fh:
            pistas = json.load(fh)["pistas"]
        for p in pistas:
            p["artista"] = artista(p["cancion"])
            p["genero"] = macro_cancion(p["etiquetas"])
    else:
        pistas = predecir(args)
    metodos = list(pistas[0]["aciertos"])
    grupos = agrupar(pistas, metodos)

    with open(path, "w") as fh:
        json.dump({"grupos": grupos, "pistas": pistas}, fh, indent=1, ensure_ascii=False)
    orden = sorted(grupos, key=lambda g: (g == "total", -grupos[g]["n_notas"]))
    print(f"{'grupo':20s}{'pistas':>7s}{'artist.':>8s}" + "".join(f"{m[:16]:>22s}" for m in metodos))
    for g in orden:
        r = grupos[g]
        celdas = "".join(f"{r['tab_acc'][m]:8.1%} [{r['ic95'][m][0]:.0%}-{r['ic95'][m][1]:.0%}]"
                         for m in metodos)
        print(f"{g:20s}{r['n_pistas']:7d}{r['n_artistas']:8d}{celdas}")
    print(f"\nDiferencia con {REF} (puntos) [IC 95 %], * = el intervalo no contiene el 0")
    otros = [m for m in metodos if m != REF]
    print(f"{'grupo':20s}" + "".join(f"{m[:16]:>24s}" for m in otros))
    for g in orden:
        d = grupos[g][f"dif_vs_{REF}"]
        celdas = ""
        for m in otros:
            lo, hi = d[m]["ic95"]
            sig = "*" if lo > 0 or hi < 0 else " "
            celdas += f"{100 * d[m]['dif']:+7.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}]{sig}"
        print(f"{g:20s}{celdas}")
    print("Guardado en", path)


def predecir(args):
    """Predice con todos los métodos y devuelve los aciertos de cada pista."""

    songs = load_tracks(os.path.join(args.data, f"{args.split}.jsonl.gz"))
    with open(os.path.join(args.dadagp, "_DadaGP_all_metadata.json")) as fh:
        meta = json.load(fh)
    etiquetas = [meta[s["meta"]["song"] + ".tokens.txt"]["genre_tokens"] for s in songs]

    fb = Fretboard()
    pitches = [[p for p, _ in s["notes"]] for s in songs]
    aciertos = {"baseline": hits_por_pista([MinFretBaseline().predict(p) for p in pitches], songs)}
    astar = AStarBurlet(fb)
    aciertos["A*"] = hits_por_pista(
        [astar.predict(p, s.get("starts")) for p, s in zip(pitches, songs)], songs)

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
        aciertos[os.path.basename(os.path.normpath(run))] = hits_por_pista([o["final"] for o in out], songs)

    metodos = list(aciertos)
    return [{"cancion": s["meta"]["song"], "artista": artista(s["meta"]["song"]),
             "etiquetas": e, "genero": macro_cancion(e), "n_notas": len(s["notes"]),
             "n_traste_5": sum(pos.fret >= 5 for pos, _ in s["tabs"]),
             "n_ataques": len(set(s["starts"])),
             "aciertos": {m: aciertos[m][i] for m in metodos}}
            for i, (s, e) in enumerate(zip(songs, etiquetas))]

if __name__ == "__main__":
    main()
