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
export const validationMetrics = Object.entries(catalog.results.variables).map(
  ([key, value]) => ({
    key,
    unit: unitLabels[value.unit] ?? value.unit,
    mae: value.metrics.mae,
    rmse: value.metrics.mean_case_seed_rmse,
  }),
);
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

// corrected_v2_downstream_v2/analysis/summary.json: equal-case/seed/horizon MAE.
export const downstreamResults = [
  {
    variable: "temperature",
    station: "T0088",
    unit: "°C",
    corrupted: 1.4029195697947963,
    giano: 1.181961197135373,
  },
  {
    variable: "humidity",
    station: "T0118",
    unit: "%",
    corrupted: 11.921975462245351,
    giano: 9.885848108908059,
  },
  {
    variable: "pressure",
    station: "T0103",
    unit: "hPa",
    corrupted: 2.6259259146181337,
    giano: 1.4358508669941243,
  },
  {
    variable: "precipitation",
    station: "T0094",
    unit: "mm",
    corrupted: 0.2915354443635459,
    giano: 0.296385198117722,
  },
  {
    variable: "wind_direction",
    station: "T0442",
    unit: "°",
    corrupted: 29.235476862135634,
    giano: 28.99114457817219,
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
