# Transcripción simbólica de guitarra: estudio comparativo MIDI → tablatura

Trabajo Fin de Máster · Máster en Ingeniería Informática · Universidad de Sevilla

Autor: Juan Ignacio García Bartolomé · Tutor: José María Luna Romera

Dada una secuencia de notas (altura y duración), el problema consiste en asignar a
cada nota una cuerda y un traste de la guitarra. Este repositorio reimplementa el
Fretting-Transformer (Hamberger et al., 2025, arXiv:2506.14223) y lo compara, con los
mismos datos y métricas, con métodos clásicos y con sus variantes GPT2 y BERT.

## Métodos

| Método | Tipo | Módulo |
|---|---|---|
| Traste mínimo (*baseline*) | Regla, sin contexto | `src/models/baseline_minfret.py` |
| A* | A-star-Guitar (Burlet y Fujinaga, 2013) con la función de coste del código de los autores ([gburlet/astar-guitar](https://github.com/gburlet/astar-guitar), MIT), sin calibrar; camino de coste mínimo sobre notas y acordes (Viterbi) | `src/models/astar_burlet.py` (`AStarBurlet`) |
| A* nota a nota (ablación) | El mismo coste sin agrupar los acordes | `src/models/astar_burlet.py` (sin instantes de inicio) |
| A* variante propia (fuera del benchmark) | Coste propio calibrado en validación; se cita como trabajo futuro | `src/models/astar.py` (`AStarChords`) |
| T5 | *Encoder-decoder*, traducción MIDI → tablatura | `src/models/t5_fretting.py` |
| GPT2 | Solo *decoder*, completar texto | `src/models/gpt2_fretting.py` |
| BERT | Solo *encoder*, rellenar huecos enmascarados | `src/models/bert_fretting.py` |

Los modelos *transformer* se entrenan desde cero con Adafactor y parada temprana.
En inferencia procesan trozos de 20 notas con una nota de contexto y su salida se
corrige con el post-procesado del artículo (`src/postprocess.py`) o con
decodificación restringida por altura (`src/models/constrained.py`).

## Estructura del código

```
src/
  fretboard.py          Mástil: afinación, posiciones candidatas de cada nota
  tokenizer_v3.py       Codificación v3 del artículo (NOTE_ON/TIME_SHIFT/NOTE_OFF → TAB/TIME_SHIFT)
  postprocess.py        Post-procesado del artículo: ventana ±5 y posición de reserva
  metrics.py            Tab accuracy, pitch accuracy y difficulty score
  data/
    dadagp_gp.py        Lectura de DadaGP (Guitar Pro): filtro de pistas acústicas, deduplicado, reparto
    guitarset.py        Lectura de GuitarSet (JAMS) y reparto por progresión de acordes
    gaps.py             Lectura de GAPS (MusicXML) y control de obras repetidas con DadaGP
    chunking.py         Troceo en ventanas de entrenamiento y en trozos de inferencia
    generos.py          Agrupación de las etiquetas de género de DadaGP en macrogéneros
    synthetic.py        Datos sintéticos, solo para las pruebas automáticas
  models/               Métodos de la tabla anterior
scripts/
  build_acoustic_index.py   Índice de pistas acústicas de DadaGP
  build_paper_dataset.py    Conjunto de DadaGP de la réplica (data/paper)
  build_mixed_dataset.py    Conjunto mezclado DadaGP + GuitarSet + GAPS (data/mixed)
  build_guitarset.py        GuitarSet y DadaGP + GuitarSet (experimentos previos)
  build_gaps.py             GAPS como test adicional (experimentos previos)
  train.py                  Entrenamiento de T5, GPT2 o BERT (--arch)
  calibrar_astar.py         Calibración de la variante propia de A* en validación
  evaluate_benchmark.py     Evaluación de todos los métodos con las métricas del artículo
  eval_por_estilo.py        Acierto por estilo y por parte en GuitarSet
  eval_por_genero.py        Acierto por macrogénero en DadaGP, con intervalos por bootstrap
  analisis_acordes.py       Acierto según el tamaño del acorde en DadaGP
  simulate_postprocess.py   Simulación que justifica la variante de post-procesado
  reproducir.sh             Reproduce los resultados finales de principio a fin
tests/                  Pruebas automáticas (pytest)
experiments/            Resultados de evaluación (JSON) e historiales de entrenamiento
```

## Datos

Los conjuntos de datos no se incluyen por sus licencias. Deben colocarse junto a la
carpeta del proyecto:

| Conjunto | Origen | Licencia | Carpeta |
|---|---|---|---|
| DadaGP v1.1 (Sarmento et al., 2021) | Bajo petición a sus autores | Solo investigación | `../DadaGP-v1.1/` |
| GuitarSet (Xi et al., 2018) | Zenodo 3371780 (anotaciones) | CC BY 4.0 | `../GuitarSet-1.1/` |
| GAPS v1 (Riley et al., 2024) | Zenodo 13962272 (sin audio) | CC BY-NC-SA 4.0 | `../GAPS-v1/` |

El conjunto mezclado (`data/mixed`) reparte cada fuente 80/10/10 por la unidad que
evita fugas entre entrenamiento y test: el fichero en DadaGP, la progresión de acordes
en GuitarSet y la obra en GAPS. En GuitarSet, la validación y el test toman una única
progresión de estilos distintos para que ningún estilo se quede sin entrenamiento: el
test contiene Funk1 y Rock1, y la validación Jazz1 y SS3.

| | Pistas de DadaGP | Fragmentos de GuitarSet | Obras de GAPS |
|---|---|---|---|
| Entrenamiento | 3.045 | 264 | 187 |
| Validación | 378 | 48 | 16 |
| Test | 372 | 48 | 18 |

## Instalación y uso

```bash
python3.12 -m pip install -r requirements.txt
python3.12 -m pytest tests/ -q          # las pruebas que necesitan DadaGP se omiten si no está
scripts/reproducir.sh                   # datos + calibración + entrenamiento + evaluación (unas 14 h en un M1 Pro)
```

## Reproducibilidad

- `scripts/reproducir.sh` contiene todos los parámetros con los que se obtuvieron los
  resultados finales. La configuración de cada entrenamiento queda además guardada en
  `experiments/entrenamiento_*.history.json`.
- Semillas fijas: 42 para los repartos de los datos y para el entrenamiento
  (inicialización de los modelos y orden de los lotes), y 0 para el *bootstrap* del
  análisis por género.
- El entrenamiento en GPU no es determinista bit a bit, por lo que un nuevo
  entrenamiento puede dar cifras ligeramente distintas. La evaluación sí lo es.
- Versiones exactas en `requirements.txt` (Python 3.12.2, PyTorch 2.14.0,
  Transformers 5.17.0, PyGuitarPro 0.11, NumPy 2.5.3, pytest 9.1.1).
- Los modelos entrenados no se incluyen por su tamaño. Se regeneran con
  `scripts/reproducir.sh`.

## Resultados guardados (`experiments/`)

| Ficheros | Contenido |
|---|---|
| `definitivo_{test,dadagp,guitarset,gaps}.eval.json` | Evaluación final en test: todas las métricas, por zona del mástil y por etapa de post-procesado |
| `definitivo_guitarset_por_estilo.json` | Acierto por estilo y por parte en el test de GuitarSet |
| `definitivo_dadagp_por_genero.json` | Acierto por macrogénero en el test de DadaGP (IC 95 % por artista y diferencias con A*) |
| `entrenamiento_*.history.json`, `.train.log` | Configuración y curvas de pérdida de los cuatro modelos finales |
| `definitivo_analisis_acordes_{dadagp,gaps}.json` | Acierto según el tamaño del evento (nota suelta o acorde), digitaciones posibles por acorde y uso de cuerdas al aire en los acordes |
| `calibracion_astar.json` | Rejilla de calibración de la variante propia de A* |
| `calibracion_replica_val.eval.json` | *Baseline* en la validación de la réplica (comparación con el artículo) |
| `calibracion_replica_val_clasicos.eval.json` | Métodos clásicos actuales en la validación de la réplica (comparación orientativa del A* original con el del artículo) |
| `previo_*` | Experimentos previos: modelos entrenados con DadaGP y GuitarSet, T5 monofónico y comparación de datos separados y mezclados. Su «A* acordes» es la variante propia con `w_high = 0,1`, la calibración de entonces; `previo_clasicos_*` evalúa en esos mismos tests los métodos clásicos actuales |

## Resultados

*Tab accuracy* en test, con los modelos entrenados con el conjunto mezclado
(`experiments/definitivo_*.eval.json`):

| Método | DadaGP | GuitarSet | GAPS | Total |
|---|---|---|---|---|
| Traste mínimo | 80,45 % | 46,48 % | 70,85 % | 79,34 % |
| A* nota a nota | 77,12 % | 61,71 % | 72,43 % | 76,60 % |
| A* | 82,84 % | 73,32 % | 69,48 % | 82,10 % |
| T5 | **89,08 %** | 74,85 % | **76,85 %** | **88,28 %** |
| GPT2 (restringida) | 88,31 % | **75,46 %** | 75,41 % | 87,51 % |
| BERT (11,2 M) | 86,73 % | 73,66 % | 75,14 % | 85,98 % |
| BERT (2,8 M) | 87,26 % | 75,33 % | 74,70 % | 86,50 % |

La variante propia de A*, fuera del benchmark, obtiene 87,77 % / 71,95 % / 76,45 % / 86,98 %.
En la columna «Total», DadaGP aporta el 93,8 % de las notas.

## Referencias principales

- Hamberger, A. et al. (2025). *Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription*. arXiv:2506.14223.
- Edwards, D. et al. (2024). *MIDI-to-Tab: Guitar Tablature Inference via Masked Language Modeling*. ISMIR. arXiv:2408.05024.
- Burlet, G. y Fujinaga, I. (2013). *Robotaba Guitar Tablature Transcription Framework*. ISMIR.
- Hsu, T.-K. et al. (2026). *Explicit Note-Event Tokenization and Pitch-Validity Constrained Decoding for MIDI-to-Tablature Transcription*. arXiv:2607.26440.
