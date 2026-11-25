import catalog from "../../generated/catalog.json";

export type Station = (typeof catalog.stations)[number];
export type StationVariable = Station["variables"][number];
export const stations = catalog.stations;
export const unitLabels: Record<string, string> = {
  degC: "°C",
  mm: "mm",
  "%": "%",
  hPa: "hPa",
  "m/s": "m/s",
  degrees: "°",
  deg: "°",
};
// Hour-weighted MAE from station-context-scores.csv and its paired controls in release-comparison.csv.
export const reconstructionResults = [
  {
    key: "temperature",
    unit: "°C",
    mae: 1.8943192,
    bilstm: 4.69895384,
    firstRelease: 4.67991046,
  },
  {
    key: "humidity",
    unit: "%",
    mae: 12.57287779,
    bilstm: 18.7261954,
    firstRelease: 18.55045027,
  },
  {
    key: "pressure",
    unit: "hPa",
    mae: 4.35757729,
    bilstm: 6.0438046,
    firstRelease: 5.98686316,
  },
  {
    key: "precipitation",
    unit: "mm",
    mae: 0.11226068,
    bilstm: 0.19171441,
    firstRelease: 0.19008985,
  },
];
export const validationInfo = {
  seeds: catalog.results.seeds,
  caseSeedGroups:
    catalog.results.variables.temperature.metrics.case_seed_groups,
  builtAt: catalog.builtAt.slice(0, 10),
};
export const catalogCounts = {
  stations: catalog.counts.stations,
  stationsWithHistory: catalog.historyCounts.stations,
  series: catalog.historyCounts.series,
  stationsWithValidation: catalog.counts.stations_with_giano_results,
  examples: catalog.counts.examples,
  stationsWithExamples: catalog.counts.stations_with_examples,
  reconstructedHours: catalog.historyCounts.reconstructed_hours,
  unfilledHours: catalog.historyCounts.unfilled_hours,
  catalogOnly: catalog.counts.stations - catalog.historyCounts.stations,
};
export const attribution = catalog.attribution;
export const team = [
  "Anita Cappello",
  "Gabriele Mininni",
  "Martina Pellegrini",
  "Daniele Corn",
  "Michele Pio Lomartire",
];
export const webvalleyTutor = "Rishabh Wanjari";
export const GITHUB_URL = "https://github.com/itsminni/giano";
export const DATASET_URL =
  "https://drive.google.com/drive/folders/1DPsYf6dVRtCqxE3mDYAx4evTj0WASXK7?usp=sharing";

// Paired forecast MAE from examples/ridge/context-scores.csv, all masks and horizons.
export const extendedRepairResults = [
  {
    variable: "temperature",
    station: "T0088",
    unit: "°C",
    linear: 1.345679,
    giano: 1.163786,
  },
  {
    variable: "humidity",
    station: "T0118",
    unit: "%",
    linear: 11.645063,
    giano: 10.020116,
  },
  {
    variable: "pressure",
    station: "T0103",
    unit: "hPa",
    linear: 1.693211,
    giano: 1.591477,
  },
  {
    variable: "precipitation",
    station: "T0094",
    unit: "mm",
    linear: 0.301833,
    giano: 0.296071,
  },
  {
    variable: "wind_direction",
    station: "T0442",
    unit: "°",
    linear: 28.94091,
    giano: 29.807296,
  },
];

export const demoSeries = Array.from({ length: 48 }, (_, t) => {
  const truth = +(
    10 +
    3.4 * Math.sin(((t - 9) / 24) * Math.PI) +
    Math.sin(t * 1.1) * 0.45
  ).toFixed(1);
  const era5 = +(10.6 + 2.3 * Math.sin(((t - 7) / 24) * Math.PI)).toFixed(1);
  return { t, osservato: t >= 16 && t <= 30 ? null : truth, vero: truth, era5 };
});
