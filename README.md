# Giano

[English](#english) · [Italiano](#italiano) · [Website](https://itsminni.github.io/giano/) · [Wiki](https://github.com/itsminni/giano/wiki)

## English

### The project

Giano reconstructs missing observations in weather station records. The project
grew out of the work started at [WebValley 2025](https://webvalley.fbk.eu/challenge/),
Fondazione Bruno Kessler's summer school, using Meteotrentino data.

The model adapts [ImputeFormer](https://arxiv.org/abs/2312.01728) to meteorological
networks, using station coordinates to represent their spatial relationships.
Separate weights cover temperature, precipitation, humidity, pressure, wind
speed and wind direction. It combines temporal interpolation with a learned
correction for longer gaps, using overlapping 72-hour windows. Existing
observations stay unchanged, gaps with insufficient context can remain empty.

The repository includes training, reconstruction, comparisons with interpolation
and BiLSTM, and an experiment on the effect of reconstruction on subsequent
forecasts. Results and methodology are described in
[Results and Evidence](https://github.com/itsminni/giano/wiki/Results-and-Evidence).

### Installation

From the repository root, install the Python environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --python 3.11 --inexact
uv run --locked pytest -q
```

Python 3.11 is the reference environment. Training and evaluation support Windows,
macOS and Linux. On Windows, use PowerShell; NVIDIA GPU training requires a
CUDA-enabled [PyTorch installation](https://pytorch.org/get-started/locally/).

### Data

The dataset is distributed through Google Drive, with the same `data/` layout
as this repository. Extract its contents into the repository's `data/` folder,
keeping the subdirectories in place.

**Download:** [Dataset on Google Drive](https://drive.google.com/drive/folders/1DPsYf6dVRtCqxE3mDYAx4evTj0WASXK7?usp=sharing).

`data/1-raw/` contains the source files; `data/2-processed-v2/` contains the hourly
data used by the release.

For pretrained inference and notebooks, download `giano-1.0.0-artifacts.zip`
from the [v1.0.0 release assets](https://github.com/itsminni/giano/releases/tag/v1.0.0).
Extract its `checkpoints/` and `artifacts/` folders into the repository root,
not into `data/`. It contains Giano and BiLSTM weights, evaluation results,
training records and website exports; source code and the dataset are separate.
[release.yaml](release.yaml) lists the release paths. See
[Release and Reproduction](https://github.com/itsminni/giano/wiki/Release-and-Reproduction)
for the complete file layout.

### Reconstruction and training

With the trained weights in place, reconstruct temperature gaps with:

```bash
uv run --locked giano-impute --data-dir data/2-processed-v2 --variables temperature --checkpoint-dir checkpoints/corrected_v2_spectral/imputeformer --model-family imputeformer --seed 42 --fallback none --output-dir artifacts/imputed/my-temperature-run
```

To train a new model instead:

```bash
uv run --locked giano-train-imputeformer --config config.yaml --data-dir data/2-processed-v2 --variables temperature --checkpoint-dir checkpoints/my-run/imputeformer --seeds 42 --resume
```

Replace `temperature` with another variable, or list several after `--variables`.
The [training guide](https://github.com/itsminni/giano/wiki/Training-and-Checkpoints)
covers configuration, multiple seeds and checkpoint resume. The `giano` command
runs the full preprocessing and training pipeline.

### Other datasets

Provide observations, station coordinates and a YAML file mapping columns, units,
timestamps and quality codes. Import and validate the dataset with:

```bash
uv run --locked giano-dataset import --config examples/datasets/import.yaml --output-dir data/external/my-network/processed
uv run --locked giano-dataset validate --data-dir data/external/my-network/processed
```

The command uses the small [example dataset](examples/datasets/). Adapt its YAML
mapping to your files. The
[dataset guide](https://github.com/itsminni/giano/wiki/Portability-to-New-Datasets)
explains the format, inference, evaluation and training on another network.

### Notebooks

```bash
uv sync --locked --group notebooks --inexact
uv run --locked --group notebooks jupyter lab
```

All notebooks are in English.

| Notebook | Contents |
|---|---|
| [01 — Data analysis](notebooks/01_data_analysis.ipynb) | Observations, coverage and missing data |
| [02 — Reconstruction diagnostics](notebooks/02_reconstruction_diagnostics.ipynb) | Model comparisons on the common benchmark |
| [03 — Downstream utility](notebooks/03_downstream_utility.ipynb) | Effect of reconstruction on subsequent forecasts |
| [04 — Reconstruction example](notebooks/04_reconstruction_example.ipynb) | A random window with observations, mask and reconstruction |

### Website and repository

The [website](https://itsminni.github.io/giano/) presents the project and
lets visitors explore stations, reconstruction examples and historical series.
It displays precomputed results. The
[website README](website/giano-weather-vision/README.md) explains how to run it
locally and edit its Italian and English text.
It also describes publishing the static version to GitHub Pages, including
losslessly compressed historical series.

- `src/giano/model/`: Giano model and training.
- `src/giano/baselines/`: BiLSTM models.
- `src/giano/downstream/`: Ridge forecaster.
- `src/giano/evaluation/`: benchmarks and comparisons.
- `tests/`: tests grouped by data, models, training, evaluation, inference and delivery.
- `tools/`: data preparation, checks and release exports.

The Wiki covers [architecture](https://github.com/itsminni/giano/wiki/Giano-Architecture),
[evaluation](https://github.com/itsminni/giano/wiki/Evaluation-Protocol),
[data format](https://github.com/itsminni/giano/wiki/Data-Contract) and
[website integration](https://github.com/itsminni/giano/wiki/Website-Station-Page).

### Team

Anita Cappello, Gabriele Mininni, Martina Pellegrini, Daniele Corn and
Michele Pio Lomartire.

Rishabh Wanjari was our main point of contact among the WebValley tutors.
The [website](https://itsminni.github.io/giano/?tab=storia)
tells the project's story and includes acknowledgements.

### References

- Nie, T. et al. (2024). *ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation*. KDD. [Paper](https://arxiv.org/abs/2312.01728) · [DOI](https://doi.org/10.1145/3637528.3671751).
- Hersbach, H. et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146, 1999–2049. [DOI](https://doi.org/10.1002/qj.3803).
- [Meteotrentino — Provincia autonoma di Trento](https://www.meteotrentino.it/): source of the station observations.
- [WebValley — FBK](https://webvalley.fbk.eu/): the summer school where the work began. [2025 challenge](https://webvalley.fbk.eu/challenge/).

## Italiano

### Il progetto

Giano ricostruisce le osservazioni mancanti nelle serie delle stazioni meteo.
Il progetto nasce dal lavoro avviato a [WebValley 2025](https://webvalley.fbk.eu/challenge/),
la scuola estiva della Fondazione Bruno Kessler, sui dati Meteotrentino.

Il modello adatta [ImputeFormer](https://arxiv.org/abs/2312.01728) alle reti
meteorologiche, usando le coordinate delle stazioni per rappresentarne le
relazioni spaziali. Utilizza pesi separati per temperatura, precipitazione,
umidità, pressione, velocità e direzione del vento. Combina l'interpolazione
temporale con una correzione appresa per le lacune più lunghe, lavorando su
finestre sovrapposte di 72 ore. Le osservazioni presenti restano inalterate.
Le lacune con contesto insufficiente possono rimanere vuote.

Il repository comprende training, ricostruzione, confronti con interpolazione e
BiLSTM e un esperimento sull'effetto della ricostruzione nelle previsioni
successive. Risultati e metodologia sono descritti in
[Risultati di ricostruzione](https://github.com/itsminni/giano/wiki/Results-and-Evidence#italiano).

### Installazione

Dalla cartella principale della repository, installare l'ambiente Python con
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --python 3.11 --inexact
uv run --locked pytest -q
```

Python 3.11 è l'ambiente di riferimento. Training e valutazione supportano
Windows, macOS e Linux. Su Windows utilizzare PowerShell; per il training su
GPU NVIDIA serve un'[installazione di PyTorch con CUDA](https://pytorch.org/get-started/locally/).

### Dati

Il dataset è distribuito tramite Google Drive e mantiene la stessa struttura
di `data/` presente nella repository. Estrarre il contenuto nella cartella
`data/` del progetto, conservando le sottocartelle.

**Download:** [Dataset su Google Drive](https://drive.google.com/drive/folders/1DPsYf6dVRtCqxE3mDYAx4evTj0WASXK7?usp=sharing).

`data/1-raw/` contiene i file sorgenti; `data/2-processed-v2/` contiene i dati
orari usati dalla release.

Per usare i pesi addestrati e i notebook, scaricare `giano-1.0.0-artifacts.zip`
dagli [allegati della release v1.0.0](https://github.com/itsminni/giano/releases/tag/v1.0.0).
Estrarre le cartelle `checkpoints/` e `artifacts/` nella radice della repository,
non dentro `data/`. Il pacchetto contiene pesi Giano e BiLSTM, risultati,
registri del training ed esportazioni per il sito; codice e dataset sono separati.
[release.yaml](release.yaml) elenca i percorsi della release. La pagina
[File e riproduzione](https://github.com/itsminni/giano/wiki/Release-and-Reproduction#italiano)
descrive la struttura completa dei file.

### Ricostruzione e training

Con i pesi addestrati al loro posto, ricostruire le lacune della temperatura con:

```bash
uv run --locked giano-impute --data-dir data/2-processed-v2 --variables temperature --checkpoint-dir checkpoints/corrected_v2_spectral/imputeformer --model-family imputeformer --seed 42 --fallback none --output-dir artifacts/imputed/my-temperature-run
```

Per addestrare invece un nuovo modello:

```bash
uv run --locked giano-train-imputeformer --config config.yaml --data-dir data/2-processed-v2 --variables temperature --checkpoint-dir checkpoints/my-run/imputeformer --seeds 42 --resume
```

Sostituire `temperature` con un'altra variabile, oppure elencarne più di una dopo
`--variables`. La [guida al training](https://github.com/itsminni/giano/wiki/Training-and-Checkpoints#italiano)
descrive configurazione, seed multipli e ripresa dai checkpoint. Il comando
`giano` esegue l'intera pipeline di preparazione dei dati e training.

### Altri dataset

Preparare osservazioni, coordinate delle stazioni e un file YAML che descriva
colonne, unità, riferimenti temporali e codici di qualità. Importare e validare
il dataset con:

```bash
uv run --locked giano-dataset import --config examples/datasets/import.yaml --output-dir data/external/my-network/processed
uv run --locked giano-dataset validate --data-dir data/external/my-network/processed
```

Il comando usa il piccolo [dataset di esempio](examples/datasets/). Adattare
la configurazione YAML ai propri file. La
[guida ai dataset esterni](https://github.com/itsminni/giano/wiki/Portability-to-New-Datasets#italiano)
spiega formato, inferenza, valutazione e training su un'altra rete.

### Notebook

```bash
uv sync --locked --group notebooks --inexact
uv run --locked --group notebooks jupyter lab
```

Tutti i notebook sono in inglese.

| Notebook | Contenuto |
|---|---|
| [01 — Analisi dei dati](notebooks/01_data_analysis.ipynb) | Osservazioni, copertura e dati mancanti |
| [02 — Diagnostica delle ricostruzioni](notebooks/02_reconstruction_diagnostics.ipynb) | Confronti tra modelli sul benchmark comune |
| [03 — Utilità downstream](notebooks/03_downstream_utility.ipynb) | Effetto della ricostruzione sulle previsioni successive |
| [04 — Esempio di ricostruzione](notebooks/04_reconstruction_example.ipynb) | Una finestra casuale con osservazioni, maschera e ricostruzione |

### Sito e repository

Il [sito](https://itsminni.github.io/giano/) presenta il progetto e permette
di esplorare stazioni, esempi di ricostruzione e serie storiche. Mostra risultati
precalcolati. Il [README del sito](website/giano-weather-vision/README.md)
spiega come avviarlo in locale e modificarne i testi in italiano e inglese.
Descrive anche la pubblicazione della versione statica su GitHub Pages, con le
serie storiche compresse senza perdita di dati.

- `src/giano/model/`: modello Giano e training.
- `src/giano/baselines/`: modelli BiLSTM.
- `src/giano/downstream/`: modello previsivo Ridge.
- `src/giano/evaluation/`: benchmark e confronti.
- `tests/`: test suddivisi per dati, modelli, training, valutazione, inferenza e distribuzione.
- `tools/`: preparazione dei dati, verifiche ed esportazioni della release.

La Wiki approfondisce [architettura](https://github.com/itsminni/giano/wiki/Giano-Architecture#italiano),
[valutazione](https://github.com/itsminni/giano/wiki/Evaluation-Protocol#italiano),
[formato dei dati](https://github.com/itsminni/giano/wiki/Data-Contract#italiano) e
[integrazione del sito](https://github.com/itsminni/giano/wiki/Website-Station-Page#italiano).

### Team

Il team Giano è composto da Anita Cappello, Gabriele Mininni, Martina Pellegrini,
Daniele Corn e Michele Pio Lomartire.

Rishabh Wanjari è stato il nostro principale riferimento fra i tutor di WebValley.
Il [sito](https://itsminni.github.io/giano/?tab=storia)
racconta la storia del progetto e riporta i ringraziamenti.

### Riferimenti

- Nie, T. et al. (2024). *ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation*. KDD. [Articolo](https://arxiv.org/abs/2312.01728) · [DOI](https://doi.org/10.1145/3637528.3671751).
- Hersbach, H. et al. (2020). *The ERA5 global reanalysis*. Quarterly Journal of the Royal Meteorological Society, 146, 1999–2049. [DOI](https://doi.org/10.1002/qj.3803).
- [Meteotrentino — Provincia autonoma di Trento](https://www.meteotrentino.it/): fonte delle osservazioni delle stazioni.
- [WebValley — FBK](https://webvalley.fbk.eu/): la scuola estiva da cui è partito il lavoro. [Challenge 2025](https://webvalley.fbk.eu/challenge/).
