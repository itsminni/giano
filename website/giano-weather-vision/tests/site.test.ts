import { describe, expect, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gzipSync } from "node:zlib";
import { createElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LanguageProvider } from "../src/components/giano/i18n";
import { MapSection } from "../src/components/giano/MapSection";
import { Story } from "../src/components/giano/Story";
import { Impact } from "../src/components/giano/Impact";
import { Guide } from "../src/components/giano/Guide";
import { ZoomView } from "../src/components/giano/ZoomView";
import {
  ReconstructionExample,
  HistoricalSeries,
} from "../src/components/giano/StationCharts";
import {
  DATASET_URL,
  demoSeries,
  downstreamResults,
  team,
  webvalleyTutor,
} from "../src/components/giano/data";
import it from "../src/content/it";
import en from "../src/content/en";
import { renderErrorPage } from "../src/lib/error-page";
import catalog from "../src/generated/catalog.json";
import {
  assetUrl,
  loadAsset,
  examplePoints,
  hiddenRanges,
  historyPoints,
  shiftDay,
  wallTime,
  type Example,
  type HistoryChunk,
} from "../src/components/giano/series";

const root = resolve(import.meta.dir, "../public/data/giano");
const json = (path: string) =>
  JSON.parse(readFileSync(resolve(root, path), "utf8"));

test("error pages provide Italian and English messages and actions", () => {
  for (const [language, dictionary] of [
    ["it-IT,it;q=0.9", it],
    ["en-GB,en;q=0.9", en],
  ] as const) {
    const html = renderErrorPage(language);
    expect(html).toContain(`<html lang="${language.slice(0, 2)}">`);
    for (const key of [
      "error.title",
      "error.description",
      "error.retry",
      "error.home",
    ] as const) {
      expect(html).toContain(dictionary[key]);
    }
  }
  expect(renderErrorPage()).toContain(it["error.title"]);
});

function render(component: ReactNode, client = new QueryClient()) {
  const navigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { language: "it-IT", languages: ["it-IT"] },
  });
  try {
    return renderToStaticMarkup(
      createElement(QueryClientProvider, {
        client,
        children: createElement(LanguageProvider, { children: component }),
      }),
    );
  } finally {
    if (navigator) Object.defineProperty(globalThis, "navigator", navigator);
    else Reflect.deleteProperty(globalThis, "navigator");
  }
}

describe("Station navigation and credits", () => {
  test("ERA5 is optional, off initially and never substituted for observations", () => {
    const variable = catalog.stations
      .find((s) => s.id === "T0146")!
      .variables.find((v) => v.key === "temperature")!;
    const path = variable.example!.json;
    const sample: Example = json(path);
    const client = new QueryClient();
    delete sample.series.era5_land;
    client.setQueryData(["example", path], structuredClone(sample));
    expect(
      render(createElement(ReconstructionExample, { variable }), client),
    ).not.toContain(it["chart.showEra5"]);
    sample.series.era5_land = sample.series.timestamps.map((_, i) =>
      i === 1 ? null : 0,
    );
    client.setQueryData(["example", path], sample);
    const html = render(
      createElement(ReconstructionExample, { variable }),
      client,
    );
    expect(html).toContain(it["chart.showEra5"]);
    expect(html.match(/type="checkbox"/g)).toHaveLength(3);
    expect(html.match(/type="checkbox"[^>]*checked=""/g)).toHaveLength(2);
    const points = examplePoints(sample);
    expect(points[0]!.era5).toBe(0);
    expect(points[1]!.era5).toBeNull();
    expect(points.map((p) => p.observed)).toEqual(
      sample.series.observed_with_mask,
    );
  });
  test("examples and histories offer a checked reconstruction control; truth remains example-only", () => {
    const variable = catalog.stations
      .find((s) => s.id === "T0146")!
      .variables.find((v) => v.key === "temperature")!;
    const client = new QueryClient();
    const examplePath = variable.example!.json;
    client.setQueryData(["example", examplePath], json(examplePath));
    const example = render(
      createElement(ReconstructionExample, { variable }),
      client,
    );
    expect(example).toContain(it["chart.showReconstruction"]);
    expect(example).toContain(it["chart.showTruth"]);
    expect(example.match(/type="checkbox"[^>]*checked=""/g)).toHaveLength(2);
    const index = json(variable.history!);
    client.setQueryData(["history", variable.history], index);
    const lower = wallTime(index.start.slice(0, 10));
    const paths = index.chunks
      .filter(
        (chunk: { start: string; end: string }) =>
          wallTime(chunk.start) < lower + 7 * 86400000 &&
          wallTime(chunk.end) >= lower,
      )
      .map((chunk: { path: string }) => chunk.path);
    client.setQueryData(
      ["history-chunks", paths],
      paths.map((path: string) => json(path)),
    );
    const history = render(
      createElement(HistoricalSeries, { variable }),
      client,
    );
    expect(history).toContain(it["chart.showReconstruction"]);
    expect(history).not.toContain(it["chart.showTruth"]);
    expect(history.match(/type="checkbox"[^>]*checked=""/g)).toHaveLength(1);
  });
  test("zoom controls are labelled, bounded initially and linked to a keyboard-scrollable view", () => {
    const html = render(
      createElement(ZoomView, { label: "Station chart", children: "chart" }),
    );
    expect(html).toContain('role="region"');
    expect(html).toContain('tabindex="0"');
    expect(html).toContain("aria-controls=");
    expect(html).toContain("100%");
    expect(html).toContain(it["zoom.in"]);
    expect(html).toContain(it["zoom.out"]);
    expect(html).toContain(it["zoom.reset"]);
    expect(html.match(/disabled=""/g)).toHaveLength(2);
  });
  test("team cards are static and distinguish the WebValley tutor", () => {
    const html = render(createElement(Story));
    expect(team).toHaveLength(5);
    expect(team).toContain("Gabriele Mininni");
    expect(team).not.toContain(webvalleyTutor);
    expect(html).toContain("Rishabh Wanjari");
    expect(html).toContain(it["story.tutor"]);
    expect(html.match(/<article\b/g)).toHaveLength(6);
    expect(html).not.toContain("hover:");
    expect(html).not.toContain("Domenico");
    expect(html).toContain(it["story.tutorNote"]);
    expect(html).toContain(it["story.thanksData"]);
    expect(html).toContain(it["story.thanksSchool"].replaceAll("'", "&#x27;"));
    expect(html).toContain(
      "https://magazine.fbk.eu/en/news/challenges-that-make-you-grow/",
    );
  });

  test("impact shows downstream forecast errors and a route to independent use", () => {
    const html = render(createElement(Impact, { onNavigate: () => {} }));
    expect(downstreamResults).toHaveLength(5);
    expect(
      downstreamResults
        .filter((row) => row.giano > row.corrupted)
        .map((row) => row.variable),
    ).toEqual(["precipitation"]);
    for (const row of downstreamResults) expect(html).toContain(row.station);
    expect(html).toContain(it["impact.downstreamBaseline"]);
    expect(html).toContain(
      it["impact.downstreamConclusion"].replaceAll("'", "&#x27;"),
    );
    expect(html).toContain(it["impact.useGuide"]);
    expect(render(createElement(Guide))).toContain(
      DATASET_URL.replaceAll("&", "&amp;"),
    );
  });

  test("the illustrative ERA5 curve supplies distinct values aligned with every hour", () => {
    expect(demoSeries).toHaveLength(48);
    expect(
      demoSeries.every(
        (point, i) => point.t === i && Number.isFinite(point.era5),
      ),
    ).toBe(true);
    const gap = demoSeries.filter((point) => point.osservato === null);
    expect(gap).toHaveLength(15);
    expect(gap.every((point) => point.era5 !== point.vero)).toBe(true);
  });

  test("station rows are buttons, availability is text and variable controls sit with the chart", () => {
    const html = render(
      createElement(MapSection, {
        stationId: "T0063",
        variableKey: "temperature",
        mode: "example",
        onSelection: () => {},
      }),
    );
    const list = html.match(/<ul\b[^>]*>(.*?)<\/ul>/s)?.[1] ?? "";
    expect(list.match(/<button\b/g)).toHaveLength(208);
    expect(list).toContain('aria-controls="station-details"');
    const availability = html.match(/<dl\b[^>]*>(.*?)<\/dl>/s)?.[1] ?? "";
    expect(availability).toContain("Disponibile");
    expect(availability).not.toContain("<button");
    expect(availability).not.toContain("rounded-");
    const details =
      html.match(/<section id="station-details"[^>]*>(.*?)<\/section>/s)?.[1] ??
      "";
    expect(details).toContain("<fieldset");
    expect(details).toContain(it["map.variableLabel"]);
    expect(details).toContain('id="station-chart"');
  });

  test("catalogue-only stations show unavailable data and no chart controls", () => {
    const station = catalog.stations.find((s) => !s.variables.length)!;
    expect(station).toBeDefined();
    const html = render(
      createElement(MapSection, {
        stationId: station.id,
        variableKey: "temperature",
        mode: "history",
        onSelection: () => {},
      }),
    );
    expect(html).toContain(it["map.catalogOnly"]);
    expect(html).toContain(it["map.unavailable"]);
    expect(html).not.toContain("<fieldset");
    expect(html).not.toContain('id="station-chart"');
  });

  test("both languages use the project and event names consistently", () => {
    for (const content of [it, en]) {
      expect(content["footer.badge"]).toContain("Premio GF Marilli 2026");
      expect(content["home.badge"]).toContain("WebValley");
      expect(JSON.stringify(content)).not.toMatch(
        /Web Valley|Meteo Trentino|HackersGen/,
      );
    }
  });
});

describe("Frozen research data", () => {
  test("headline metrics use the released aggregation without station averaging", () => {
    expect(catalog.results).toEqual(json("results.json"));
    expect(catalog.results.variables.temperature.metrics.mae).toBeCloseTo(
      0.6863738255,
      9,
    );
    expect(
      catalog.results.variables.temperature.metrics.mean_case_seed_rmse,
    ).toBeCloseTo(0.9945163172, 9);
  });
  test("processing, validation and examples are independent", () => {
    expect(catalog.stations).toHaveLength(208);
    expect(catalog.stations.filter((s) => s.hasHistory)).toHaveLength(171);
    expect(catalog.stations.filter((s) => s.hasValidation)).toHaveLength(84);
    expect(catalog.stations.filter((s) => s.hasExample)).toHaveLength(132);
    expect(
      catalog.stations.filter((s) => s.hasHistory && !s.hasValidation).length,
    ).toBeGreaterThan(0);
  });
  test("every station/variable points to its own available assets", () => {
    let histories = 0,
      examples = 0;
    for (const station of catalog.stations)
      for (const variable of station.variables) {
        for (const path of [variable.history, variable.example?.json]) {
          if (!path) continue;
          const data = json(path);
          expect(data.station).toBe(station.id);
          expect(data.variable).toBe(variable.key);
          expect(assetUrl(path)).toBe(`/data/giano/${path}`);
        }
        if (variable.history) histories++;
        if (variable.example) examples++;
        const original = json(`stations/${station.id}.json`).variables[
          variable.key
        ].giano_results;
        if (!original) expect(variable.metrics).toBeNull();
        else {
          expect(variable.metrics?.mae).toBe(original.metrics.mae);
          expect(variable.metrics?.rmse).toBe(
            original.metrics.mean_case_seed_rmse,
          );
        }
      }
    expect(histories).toBe(497);
    expect(examples).toBe(369);
  });
  test("the real example never substitutes truth for the reconstruction", () => {
    const example = json("examples/T0063/temperature.json") as Example;
    const points = examplePoints(example);
    expect(points).toHaveLength(72);
    expect(points.filter((p) => p.hidden !== null)).toHaveLength(
      example.metrics.hidden_points,
    );
    expect(hiddenRanges(example)).toHaveLength(1);
    expect(
      points.some((p) => p.hidden !== null && p.reconstructed !== p.hidden),
    ).toBe(true);
    points.forEach((point, i) => {
      expect(point.observed).toBe(example.series.observed_with_mask[i] ?? null);
      expect(point.reconstructed).toBe(
        example.series.synthetic_mask[i]
          ? (example.series.reconstruction[i] ?? null)
          : null,
      );
    });
  });
});

describe("Time series display", () => {
  test("Pages assets use the repository prefix and compressed histories", () => {
    expect(
      assetUrl("histories/T0063/temperature/2024.json", "/giano/", true),
    ).toBe("/giano/data/giano/histories/T0063/temperature/2024.json.gz");
    expect(assetUrl("examples/T0063/temperature.json", "/giano/", true)).toBe(
      "/giano/data/giano/examples/T0063/temperature.json",
    );
    expect(assetUrl("histories/T0063/temperature/index.json", "/", false)).toBe(
      "/data/giano/histories/T0063/temperature/index.json",
    );
  });
  test("plain and gzip assets preserve zeros, nulls and precision", async () => {
    const data = {
      observed: [0, null, 0.12345678901234568],
      giano_estimates: [null, 4, null],
    };
    const json = JSON.stringify(data);
    for (const body of [
      new TextEncoder().encode(json),
      new Uint8Array(gzipSync(json)),
    ]) {
      const fetch = spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(body),
      );
      try {
        expect(
          await loadAsset<typeof data>("histories/T0063/temperature/2024.json"),
        ).toEqual(data);
      } finally {
        fetch.mockRestore();
      }
    }
  });
  test("failed asset requests remain errors", async () => {
    const fetch = spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("Missing", { status: 404 }),
    );
    try {
      await expect(
        loadAsset("histories/T0063/temperature/2024.json"),
      ).rejects.toThrow("404");
    } finally {
      fetch.mockRestore();
    }
  });
  test("year boundaries preserve hourly order, zeros and unresolved gaps", () => {
    const chunks: HistoryChunk[] = [
      {
        time_start: "2024-12-31T22:00:00",
        step_seconds: 3600,
        length: 2,
        observed: [0, null],
        giano_estimates: [null, 3],
        era5_land: [0, null],
      },
      {
        time_start: "2025-01-01T00:00:00",
        step_seconds: 3600,
        length: 2,
        observed: [null, 5],
        giano_estimates: [null, null],
      },
    ];
    const points = historyPoints(chunks, "2024-12-31", 3);
    expect(points).toEqual([
      {
        time: "2024-12-31T22:00:00",
        observed: 0,
        reconstructed: null,
        era5: 0,
      },
      {
        time: "2024-12-31T23:00:00",
        observed: null,
        reconstructed: 3,
        era5: null,
      },
      {
        time: "2025-01-01T00:00:00",
        observed: null,
        reconstructed: null,
        era5: null,
      },
      {
        time: "2025-01-01T01:00:00",
        observed: 5,
        reconstructed: null,
        era5: null,
      },
    ]);
    expect(historyPoints(chunks, "2025-01-01", 1)).toHaveLength(2);
  });
  test("source labels survive leap years and DST without a time-zone conversion", () => {
    expect(shiftDay("2024-02-28", 1)).toBe("2024-02-29");
    expect(shiftDay("2024-03-31", 1)).toBe("2024-04-01");
    expect(
      wallTime("2024-03-31T03:00:00") - wallTime("2024-03-31T02:00:00"),
    ).toBe(3600000);
  });
  test.each([
    "../release.yaml",
    "https://example.org/data.json",
    "histories/../../secret.json",
  ])("reject unrelated asset paths: %s", (path) => {
    expect(() => assetUrl(path)).toThrow();
  });
  test("Italian and English have the same keys and placeholders", () => {
    expect(Object.keys(it).sort()).toEqual(Object.keys(en).sort());
    for (const key of Object.keys(it) as (keyof typeof it)[]) {
      expect(it[key].match(/\{\w+\}/g)?.sort() ?? []).toEqual(
        en[key].match(/\{\w+\}/g)?.sort() ?? [],
      );
    }
  });
});
