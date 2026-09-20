export type Example = {
  station: string;
  variable: string;
  unit: string;
  metrics: { hidden_points: number; mae: number; rmse: number };
  series: {
    timestamps: string[];
    observed_with_mask: (number | null)[];
    ground_truth: (number | null)[];
    reconstruction: (number | null)[];
    synthetic_mask: boolean[];
    era5_land?: (number | null)[];
  };
};
export type HistoryIndex = {
  station: string;
  variable: string;
  unit: string;
  start: string;
  end: string;
  observed_hours: number;
  reconstructed_hours: number;
  unfilled_hours: number;
  chunks: { year: string; path: string; start: string; end: string }[];
  monthly_overview: {
    month: string;
    observed_hours: number;
    reconstructed_hours: number;
    unfilled_hours: number;
  }[];
};
export type HistoryChunk = {
  time_start: string;
  step_seconds: number;
  length: number;
  observed: (number | null)[];
  giano_estimates: (number | null)[];
  era5_land?: (number | null)[];
};
export type SeriesPoint = {
  time: string;
  observed: number | null;
  reconstructed: number | null;
  hidden?: number | null;
  era5?: number | null;
};

export const assetUrl = (
  path: string,
  base = import.meta.env.BASE_URL ?? "/",
  compressed = import.meta.env.MODE === "pages",
) => {
  if (
    !/^(examples|histories|images)\/[A-Za-z0-9_/-]+\.(json|png)$/.test(path) ||
    path.includes("..")
  ) {
    throw new Error("Invalid research asset path");
  }
  const suffix =
    compressed && path.startsWith("histories/") && path.endsWith(".json")
      ? ".gz"
      : "";
  return `${base}data/giano/${path}${suffix}`;
};
export async function loadAsset<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(assetUrl(path), signal ? { signal } : {});
  if (!response.ok) throw new Error(`Data request failed (${response.status})`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).json() as Promise<T>;
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

// Use UTC arithmetic without converting source timestamps.
export const wallTime = (value: string) =>
  Date.parse(`${value.slice(0, 19)}${value.length === 10 ? "T00:00:00" : ""}Z`);
export const timeLabel = (value: string) =>
  value.slice(0, 16).replace("T", " ");
export const shiftDay = (value: string, days: number) =>
  new Date(wallTime(value) + days * 86400000).toISOString().slice(0, 10);
export function examplePoints(example: Example): SeriesPoint[] {
  return example.series.timestamps.map((time, i) => ({
    time: time.slice(0, 19),
    observed: example.series.observed_with_mask[i] ?? null,
    era5: example.series.era5_land?.[i] ?? null,
    reconstructed: example.series.synthetic_mask[i]
      ? (example.series.reconstruction[i] ?? null)
      : null,
    hidden: example.series.synthetic_mask[i]
      ? (example.series.ground_truth[i] ?? null)
      : null,
  }));
}
export function hiddenRanges(example: Example) {
  const ranges: { start: string; end: string }[] = [];
  let start: string | null = null;
  example.series.synthetic_mask.forEach((hidden, i, mask) => {
    if (hidden && start === null)
      start = example.series.timestamps[i]!.slice(0, 19);
    if (start !== null && (!hidden || i === mask.length - 1)) {
      ranges.push({
        start,
        end: example.series.timestamps[hidden ? i : i - 1]!.slice(0, 19),
      });
      start = null;
    }
  });
  return ranges;
}
export function historyPoints(
  chunks: HistoryChunk[],
  start: string,
  days: number,
): SeriesPoint[] {
  const lower = wallTime(start),
    upper = lower + days * 86400000;
  return chunks
    .flatMap((chunk) => {
      const origin = wallTime(chunk.time_start);
      return chunk.observed.flatMap((observed, i) => {
        const time = origin + i * chunk.step_seconds * 1000;
        if (time < lower || time >= upper) return [];
        return [
          {
            time: new Date(time).toISOString().slice(0, 19),
            observed,
            era5: chunk.era5_land?.[i] ?? null,
            reconstructed:
              observed === null ? (chunk.giano_estimates[i] ?? null) : null,
          },
        ];
      });
    })
    .sort((a, b) => a.time.localeCompare(b.time));
}
