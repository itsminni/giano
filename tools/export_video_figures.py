"""Export Italian video slides from the saved release and NOAA pilot results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from giano.provenance import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebooks.release_artifacts import load_benchmarks, load_downstream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "giano/imputeformer": ("Giano", "#007F86", "o"),
    "bilstm/fair": ("BiLSTM fair", "#3A64B8", "s"),
    "bilstm/legacy_retrained": ("BiLSTM legacy-style", "#98649A", "^"),
    "interpolation": ("Interpolazione", "#BD690C", "D"),
}
LABELS = {
    "temperature": ("Temperatura", "°C"),
    "humidity": ("Umidità", "%"),
    "pressure": ("Pressione", "hPa"),
    "wind_speed": ("Velocità del vento", "m/s"),
    "wind_direction": ("Direzione del vento", "°"),
    "precipitation": ("Precipitazione", "mm"),
}


def frame(title: str, subtitle: str):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#FAFCFD")
    fig.text(0.055, 0.91, title, fontsize=29, weight="bold", color="#102D42")
    fig.text(0.055, 0.855, subtitle, fontsize=17, color="#4C6170")
    return fig


def save(fig, output: Path, name: str, footer: str) -> None:
    fig.text(0.055, 0.055, footer, fontsize=13, color="#4C6170")
    fig.savefig(output / f"{name}.png", dpi=120)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def table(
    fig, columns: list[str], rows: list[list[str]], bounds, highlight: int = 1
) -> None:
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    grid = ax.table(
        cellText=rows, colLabels=columns, cellLoc="center", bbox=(0, 0, 1, 1)
    )
    grid.auto_set_font_size(False)
    grid.set_fontsize(17)
    for (row, col), cell in grid.get_celld().items():
        cell.set_edgecolor("#FAFCFD")
        cell.set_linewidth(3)
        cell.set_facecolor("#102D42" if row == 0 else "#EAF0F4")
        cell.set_text_props(color="white" if row == 0 else "#102D42")
        if col == highlight and row > 0:
            cell.set_text_props(weight="bold", color="#007F86")


def number(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def downstream_figures(output: Path) -> list[str]:
    fig = frame(
        "Una ricostruzione utile anche dopo?",
        "Cambia solo lo storico in ingresso, non il modello",
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    for y, label in [
        (0.63, "Storico con lacune"),
        (0.37, "Storico ricostruito\nda Giano"),
    ]:
        ax.text(
            0.18,
            y,
            label,
            ha="center",
            va="center",
            fontsize=21,
            bbox={
                "boxstyle": "round,pad=1",
                "facecolor": "#EAF0F4",
                "edgecolor": "none",
            },
        )
        ax.annotate(
            "",
            (0.40, y),
            (0.32, y),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#007F86"},
        )
        ax.text(
            0.51,
            y,
            "Stesso forecaster\nRidge, pesi fissi",
            ha="center",
            va="center",
            fontsize=20,
            bbox={
                "boxstyle": "round,pad=1",
                "facecolor": "#DCEEF0",
                "edgecolor": "none",
            },
        )
        ax.annotate(
            "",
            (0.76, y),
            (0.63, y),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#007F86"},
        )
        ax.text(
            0.85,
            y,
            "Previsioni\n↓\nErrore (MAE)",
            ha="center",
            va="center",
            fontsize=21,
            color="#102D42",
        )
    fig.text(
        0.055,
        0.17,
        "Orizzonti: 1, 6, 12 e 24 ore · confronto con i valori reali successivi",
        fontsize=19,
        color="#102D42",
    )
    save(
        fig,
        output,
        "05_downstream_workflow",
        "Il forecaster è addestrato su dati completi.",
    )

    sources, cells = [], []
    for variable in [
        "temperature",
        "humidity",
        "pressure",
        "precipitation",
        "wind_direction",
    ]:
        payload, path = load_downstream(ROOT, variable)
        sources.append(str(path))
        summaries = pd.DataFrame(payload["summaries"])
        scores = summaries.groupby("method").mae.mean()
        stations = summaries.station.unique()
        if len(stations) != 1 or set(summaries.seed) != {42, 43, 44, 45, 46}:
            raise ValueError(f"Unexpected downstream station/seeds: {variable}")
        gain = 100 * (1 - scores["giano"] / scores["corrupted"])
        cells.append(
            [
                f"{LABELS[variable][0]} ({LABELS[variable][1]})\n{stations[0]}",
                number(scores["corrupted"]),
                number(scores["giano"]),
                f"{number(gain, 1)}%",
            ]
        )
    fig = frame(
        "L'effetto sulle previsioni",
        "MAE del forecaster · storico danneggiato oppure ricostruito da Giano",
    )
    table(
        fig,
        ["Variabile · stazione", "Danneggiato", "Ricostruito", "Riduzione MAE"],
        cells,
        [0.04, 0.21, 0.92, 0.54],
        highlight=2,
    )
    fig.text(
        0.055,
        0.135,
        "La velocità del vento non aveva nessuna stazione idonea.",
        fontsize=16,
        color="#4C6170",
    )
    save(
        fig,
        output,
        "06_downstream_results",
        "Una stazione per variabile · media su 5 seed, 4 tipi di lacuna e 4 orizzonti · riduzione negativa = peggioramento",
    )
    return sources


def reconstruction_pair(output: Path) -> Path:
    source = (
        ROOT / "artifacts/website/giano-history/histories/T0146/temperature/2023.json"
    )
    history = json.loads(source.read_text())
    times = pd.date_range(history["time_start"], periods=history["length"], freq="h")
    window = (times >= pd.Timestamp("2023-01-18T12:00:00")) & (
        times < pd.Timestamp("2023-01-21T12:00:00")
    )
    observed = np.asarray(history["observed"], dtype=float)[window]
    estimates = np.asarray(history["giano_estimates"], dtype=float)[window]
    missing = ~np.isfinite(observed)
    filled = np.where(missing, estimates, observed)
    if len(observed) != 72 or not missing.any() or not np.isfinite(filled).all():
        raise ValueError("Expected a reconstructed 72-hour Aldeno example")
    # Include the observed endpoints so the reconstructed segments join the series.
    adjacent = np.r_[False, missing[:-1]] | np.r_[missing[1:], False]
    reconstruction = np.where(missing | adjacent, filled, np.nan)
    edges = np.diff(np.r_[False, missing, False].astype(int))
    gaps = list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9), dpi=120, sharex=True, sharey=True, facecolor="#FAFCFD"
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.23, top=0.77, hspace=0.34)
    fig.text(
        0.09,
        0.94,
        "Aldeno (San Zeno) · Temperatura",
        fontsize=26,
        weight="bold",
        color="#102D42",
    )
    fig.text(
        0.09,
        0.89,
        "18–21 gennaio 2023 · finestra di 72 ore · lacune reali",
        fontsize=17,
        color="#4C6170",
    )
    x = np.arange(len(observed))
    for ax, label in zip(axes, ["Serie originale", "Serie ricostruita"], strict=True):
        ax.set_title(label, loc="left", fontsize=16, color="#102D42", pad=10)
        ax.set_ylabel("Temperatura (°C)", fontsize=14, color="#4C6170")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#BCC8D0")
        ax.tick_params(colors="#4C6170", labelsize=12)
        ax.grid(axis="y", alpha=0.15)
        for start, end in gaps:
            ax.axvspan(start - 0.5, end - 0.5, color="#FFF0C2", linewidth=0)
        ax.plot(x, observed, color="#007F86", linewidth=3.5)
        ax.margins(x=0, y=0.15)
    ticks = np.arange(0, len(observed), 12)
    axes[1].set_xticks(
        ticks, [times[window][int(i)].strftime("%d/%m\n%H:%M") for i in ticks]
    )
    axes[1].set_xlabel("Data e ora", fontsize=14, color="#4C6170", labelpad=8)
    (repaired,) = axes[1].plot(
        x, reconstruction, color="#BD690C", linewidth=3.5, linestyle="--"
    )
    fig.legend(
        [
            Line2D([], [], color="#007F86", linewidth=3.5),
            Patch(color="#FFF0C2", linewidth=0),
            repaired,
        ],
        ["Dati osservati", "Dati mancanti", "Ricostruzione Giano"],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.87),
        ncol=3,
        frameon=False,
        fontsize=16,
    )
    reconstructed = int((missing & np.isfinite(estimates)).sum())
    durations = " e ".join(str(end - start) for start, end in gaps)
    fig.text(
        0.09,
        0.075,
        f"{len(gaps)} lacune ({durations} ore) · {int(missing.sum())} valori mancanti · {reconstructed} ricostruiti",
        fontsize=16,
        weight="bold",
        color="#102D42",
    )
    save(
        fig,
        output,
        "07_gap_to_analysis",
        "MAE e RMSE non disponibili sui gap reali: mancano i valori originali di confronto.",
    )
    return source


def prepare_imputation(output: Path) -> None:
    """Copy a week of real observations for a short, repeatable CLI recording."""
    import xarray as xr

    from giano.netcdf import open_dataset_robust

    stations = ["T0146", "T0368", "T0147", "T0210", "T0148", "T0454", "T0129", "T0369"]
    directory = output / "imputation/input/all"
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    for station in stations:
        paths = list(
            (ROOT / "data/2-processed-v2").glob(f"*/{station}_temperature_merged.nc")
        )
        if len(paths) != 1:
            raise ValueError(f"Expected one source file for {station}")
        source = paths[0]
        with open_dataset_robust(source) as data:
            week: xr.Dataset = data.sel(time=slice("2023-01-17", "2023-01-23")).load()
        if week.sizes["time"] != 168:
            raise ValueError(f"Incomplete recording window: {station}")
        week.to_netcdf(directory / source.name, engine="h5netcdf")
        sources[str(source.relative_to(ROOT))] = file_sha256(source)
    (output / "imputation/sources.json").write_text(
        json.dumps(
            {
                "period": ["2023-01-17T00:00:00", "2023-01-23T23:00:00"],
                "selection": "Aldeno and seven nearby stations. Existing natural gaps",
                "sources": sources,
            },
            indent=2,
        )
        + "\n"
    )


def export(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = load_benchmarks(ROOT)
    data = pd.DataFrame(rows)
    data["family"] = [
        "interpolation"
        if row["model"] == "interpolation"
        else f"{row['model']}/{row['model_variant']}"
        for row in rows
    ]
    selected = data.loc[
        (data.mask_type == "block")
        & data.mask_parameter.isin([6, 12, 24])
        & data.variable.isin(["temperature", "humidity", "pressure"])
    ]
    grouped = selected.groupby(["variable", "mask_parameter", "family"])["mae"]
    if not grouped.count().eq(5).all() or len(selected) != 180:
        raise ValueError(
            "Expected three durations, three variables, four models and five seeds"
        )
    scores = grouped.mean().unstack("family")
    score_rows = scores.to_dict(orient="index")
    scores.to_csv(output / "long_gaps_mae.csv")
    fig = frame(
        "Quando la lacuna si allunga",
        "Errore medio di ricostruzione · blocchi consecutivi di 6, 12 e 24 ore",
    )
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.25, top=0.74, wspace=0.3)
    for ax, variable in zip(axes, ["temperature", "humidity", "pressure"], strict=True):
        label, unit = LABELS[variable]
        for family, (name, color, marker) in MODELS.items():
            ax.plot(
                [6, 12, 24],
                scores.loc[variable, family],
                label=name,
                color=color,
                marker=marker,
                linewidth=3,
                markersize=8,
            )
        ax.set(
            title=label,
            ylabel=f"MAE ({unit})",
            xlabel="Durata della lacuna (ore)",
            xticks=[6, 12, 24],
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=4,
        frameon=False,
        fontsize=16,
    )
    save(
        fig,
        output,
        "01_long_gaps_trend",
        "Meteotrentino · validation · media sui 5 seed (42–46) · MAE più basso = ricostruzione migliore",
    )

    fig = frame(
        "Il confronto sulle 24 ore",
        "MAE medio · stesse osservazioni nascoste per tutti i modelli",
    )
    cells = []
    gains = []
    for variable in ["temperature", "humidity", "pressure"]:
        score = score_rows[(variable, 24)]
        gain = 100 * (1 - score["giano/imputeformer"] / score["bilstm/fair"])
        gains.append(f"{LABELS[variable][0]}: −{number(gain, 1)}%")
        cells.append(
            [f"{LABELS[variable][0]}\n({LABELS[variable][1]})"]
            + [number(score[key]) for key in MODELS]
        )
    table(
        fig,
        ["Variabile", "Giano", "BiLSTM fair", "BiLSTM\nlegacy-style", "Interpolazione"],
        cells,
        [0.05, 0.36, 0.90, 0.38],
    )
    fig.text(
        0.055,
        0.25,
        "Riduzione del MAE rispetto alla BiLSTM fair",
        fontsize=19,
        weight="bold",
        color="#102D42",
    )
    fig.text(0.055, 0.18, "     ·     ".join(gains), fontsize=21, color="#007F86")
    save(
        fig,
        output,
        "02_long_gaps_24h",
        "Meteotrentino · validation · media sui 5 seed (42–46) · percentuali calcolate sui valori non arrotondati",
    )

    csv_dir = ROOT / "data/external/noaa-isd-lite-2024/csv"
    config = csv_dir / "import.yaml"
    station_path = csv_dir / "stations.csv"
    benchmark = ROOT / "artifacts/evaluation/external_noaa_2024/benchmark.json"
    pilot = json.loads(benchmark.read_text())
    protocol = pilot["protocol"]
    if (
        protocol["seed_mode"] != "mask_only"
        or protocol["seeds"] != [42, 43, 44]
        or protocol["split"] != "val"
    ):
        raise ValueError("Unexpected NOAA pilot protocol")
    if any(
        c["training_config"]["seed"] != 42 for c in protocol["checkpoints"].values()
    ):
        raise ValueError("Expected seed-42 pretrained weights")
    stations = pd.read_csv(station_path, dtype={"station_id": str})
    expected = ["LOCARNO-MONTI", "MAGADINO-CADENAZZO", "LUGANO"]
    if stations["name"].tolist() != expected:
        raise ValueError("Unexpected pilot stations")
    fig = frame(
        "Una nuova rete, lo stesso importatore",
        "NOAA ISD-Lite · Svizzera · osservazioni orarie del 2024",
    )
    fig.text(
        0.055,
        0.75,
        "data/external/noaa-isd-lite-2024/",
        fontsize=18,
        weight="bold",
        color="#102D42",
    )
    fig.text(
        0.065,
        0.68,
        "csv/\n    observations.csv\n    stations.csv\n    import.yaml\nprocessed/\n    all/\n    manifest.json",
        va="top",
        fontsize=19,
        fontfamily="monospace",
        linespacing=1.55,
    )
    fig.text(0.56, 0.75, "import.yaml", fontsize=18, weight="bold", color="#102D42")
    fig.text(
        0.56,
        0.69,
        config.read_text().strip(),
        va="top",
        fontsize=16,
        fontfamily="monospace",
        linespacing=1.15,
    )
    fig.text(
        0.055,
        0.16,
        "giano-dataset import --config …/csv/import.yaml --output-dir …/processed",
        fontsize=15,
        fontfamily="monospace",
        color="#007F86",
    )
    save(
        fig,
        output,
        "03_noaa_import",
        "Configurazione del pilot · coordinate delle stazioni in stations.csv · timestamp UTC",
    )

    external = pd.DataFrame(pilot["results"])
    counts = external.groupby(["variable", "model"])["mae"].count()
    if (
        len(counts) != 6
        or not counts.eq(39).all()
        or not np.isfinite(external.mae).all()
    ):
        raise ValueError(
            "Expected three variables, two models, 13 cases and three mask seeds"
        )
    means = external.groupby(["variable", "model"])["mae"].mean().unstack("model")
    means.to_csv(output / "noaa_pilot_mae.csv")
    pilot_rows = means.to_dict(orient="index")
    fig = frame(
        "Il pilot su una rete esterna",
        "Pesi Giano seed 42 già addestrati",
    )
    for x, station in zip(
        [0.055, 0.375, 0.695], stations.itertuples(index=False), strict=True
    ):
        fig.text(
            x,
            0.72,
            str(station.name).title(),
            fontsize=22,
            weight="bold",
            color="#102D42",
        )
        fig.text(x, 0.665, str(station.station_id), fontsize=16, color="#4C6170")
    cells = []
    for variable in ["temperature", "wind_speed", "wind_direction"]:
        pilot_score = pilot_rows[variable]
        gain = 100 * (1 - pilot_score["giano"] / pilot_score["interpolation"])
        delta = f"{number(gain, 1)}%" if gain >= 0 else f"−{number(abs(gain), 2)}%"
        cells.append(
            [
                f"{LABELS[variable][0]}\n({LABELS[variable][1]})",
                number(pilot_score["giano"]),
                number(pilot_score["interpolation"]),
                delta,
            ]
        )
    table(
        fig,
        ["Variabile", "MAE Giano", "MAE interpolazione", "Riduzione MAE"],
        cells,
        [0.05, 0.22, 0.90, 0.37],
    )
    fig.text(
        0.055,
        0.145,
        "Miglioramento su temperatura e velocità del vento; direzione sostanzialmente invariata.",
        fontsize=17,
        color="#102D42",
    )
    save(
        fig,
        output,
        "04_noaa_pilot",
        "NOAA ISD-Lite 2024 · validation · media su 13 casi e 3 seed di mascheramento (42–44) · 3 stazioni",
    )
    sources.extend([str(config), str(station_path), str(benchmark)])
    sources.extend(downstream_figures(output))
    sources.append(str(reconstruction_pair(output)))
    (output / "sources.json").write_text(
        json.dumps(
            {
                "aggregation": "Arithmetic mean of recorded case/seed MAE",
                "sources": {
                    str(Path(path).relative_to(ROOT)): file_sha256(Path(path))
                    for path in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported seven PNG/SVG pairs and source tables to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/video")
    parser.add_argument(
        "--prepare-imputation",
        action="store_true",
        help="Copy one week from eight stations for the CLI recording",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 16,
        }
    )
    export(args.output_dir)
    if args.prepare_imputation:
        prepare_imputation(args.output_dir)


if __name__ == "__main__":
    main()
