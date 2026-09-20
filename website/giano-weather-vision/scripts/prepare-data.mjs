import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const site = fileURLToPath(new URL("../", import.meta.url));
const source = resolve(
  process.env.GIANO_SITE_DATA ??
    resolve(site, "../../artifacts/website/giano-history"),
);
const read = async (path) =>
  JSON.parse(await readFile(resolve(source, path), "utf8"));
const manifest = await read("manifest.json");
if (
  manifest.kind !== "giano_station_presentation" ||
  manifest.history_schema_version !== 1
) {
  throw new Error(
    "Use the complete Giano history export, including station examples.",
  );
}
const destination = resolve(site, "public/data/giano");
if (source === destination)
  throw new Error("Source and generated output must be separate.");

// Verify source hashes before copying.
for (const [path, expected] of Object.entries(manifest.files)) {
  const absolute = resolve(source, path);
  if (!absolute.startsWith(`${source}/`))
    throw new Error(`Invalid asset path: ${path}`);
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(absolute)) hash.update(chunk);
  if (hash.digest("hex") !== expected)
    throw new Error(`Asset checksum mismatch: ${path}`);
}
const geo = await read("stations.geojson");
const variables = [
  "temperature",
  "precipitation",
  "humidity",
  "pressure",
  "wind_speed",
  "wind_direction",
];
const stations = await Promise.all(
  geo.features.map(async (feature) => {
    const card = await read(feature.properties.card);
    return {
      id: card.id,
      name: card.name,
      lon: card.geometry.coordinates[0],
      lat: card.geometry.coordinates[1],
      elevation: card.elevation_m,
      startDate: card.provider_start_date,
      endDate: card.provider_end_date,
      status: card.provider_status,
      hasData: card.has_local_data,
      hasHistory: card.has_history,
      hasExample: card.has_example,
      hasValidation: card.has_giano_results,
      variables: variables
        .filter((key) => card.variables[key])
        .map((key) => {
          const value = card.variables[key];
          const coverage = value.data_coverage;
          const results = value.giano_results;
          return {
            key,
            unit: value.unit,
            observedHours: coverage?.observed_hours ?? null,
            missingHours:
              coverage?.natural_missing_hours_inside_coverage ?? null,
            first: coverage?.first_observation ?? null,
            last: coverage?.last_observation ?? null,
            history: value.history ?? null,
            example: value.example ?? null,
            metrics: results
              ? {
                  mae: results.metrics.mae,
                  rmse: results.metrics.mean_case_seed_rmse,
                  groups: results.available_case_seed_groups,
                  expectedGroups: results.expected_case_seed_groups,
                }
              : null,
          };
        }),
    };
  }),
);
stations.sort((a, b) => a.name.localeCompare(b.name, "it"));
const results = await read("results.json");
const catalog = {
  stations,
  results,
  counts: manifest.counts,
  historyCounts: manifest.history_counts,
  builtAt: manifest.built_at,
  attribution: manifest.station_catalog.attribution,
};
await mkdir(resolve(site, "src/generated"), { recursive: true });
await writeFile(
  resolve(site, "src/generated/catalog.json"),
  JSON.stringify(catalog),
);
await mkdir(resolve(site, "public/data"), { recursive: true });
await cp(source, destination, {
  recursive: true,
  mode: constants.COPYFILE_FICLONE,
});
console.log(
  `Prepared ${stations.length} stations, ${manifest.history_counts.series} histories and ${manifest.counts.examples} examples.`,
);
