#!/usr/bin/env bash
# Reproduce los resultados finales: construye los datos, entrena los cuatro
# modelos y evalúa todos los métodos en cada conjunto de test.
#
# Requiere los conjuntos originales junto a esta carpeta (ver README):
#   ../DadaGP-v1.1/   ../GuitarSet-1.1/   ../GAPS-v1/
# En un Mac conviene lanzarlo con `caffeinate -i scripts/reproducir.sh` para que
# no se suspenda (el entrenamiento completo dura unas 14 horas en un M1 Pro).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}

# 1. Datos
$PY scripts/build_acoustic_index.py
$PY scripts/build_paper_dataset.py
$PY scripts/build_mixed_dataset.py

# 2. Calibración de la variante propia de A*, fuera del benchmark (sus valores por
#    defecto ya son el resultado; el A* del benchmark usa los pesos del original)
$PY scripts/calibrar_astar.py

# 3. Entrenamiento (Adafactor, parada temprana con paciencia 5)
train() {
  local run=$1; shift
  mkdir -p "runs/$run"
  $PY scripts/train.py --data data/mixed --durations-from-data --epochs 100 \
      --eval-tracks 200 --out "runs/$run" "$@" | tee "runs/$run/train.log"
  cp "runs/$run/history.json" "experiments/entrenamiento_${run#mixed_}.history.json"
}
train mixed_t5           --arch t5   --max-notes 170
train mixed_gpt2         --arch gpt2 --max-notes 100
train mixed_bert_grande  --arch bert --max-notes 127
train mixed_bert_pequeno --arch bert --max-notes 127 \
  --model-config '{"hidden_size": 128, "num_hidden_layers": 8, "num_attention_heads": 4, "intermediate_size": 1024}'

# 4. Evaluación en test (global y por origen de los datos)
RUNS=(--run runs/mixed_t5 --run runs/mixed_gpt2 --run runs/mixed_bert_grande --run runs/mixed_bert_pequeno)
for split in test test_dadagp test_guitarset test_gaps; do
  $PY scripts/evaluate_benchmark.py --data data/mixed --split "$split" "${RUNS[@]}" \
      --constrained --name "definitivo_${split#test_}"
done
$PY scripts/eval_por_estilo.py --data data/mixed --split test_guitarset "${RUNS[@]}" \
    --name definitivo_guitarset_por_estilo
$PY scripts/eval_por_genero.py --data data/mixed "${RUNS[@]}" \
    --name definitivo_dadagp_por_genero
for split in test_dadagp test_gaps; do
  $PY scripts/analisis_acordes.py --data data/mixed --split "$split"
done
