import { useState } from "react";
import { Check, ChevronRight, MapPin, Search } from "lucide-react";
import {
  attribution,
  catalogCounts,
  stations,
  unitLabels,
  type Station,
} from "./data";
import { useI18n } from "./i18n";
import { TrentinoMap } from "./TrentinoMap";
import { HistoricalSeries, ReconstructionExample } from "./StationCharts";

export type ViewMode = "example" | "history";
type Filter = "all" | "history" | "validation" | "examples";
export function MapSection({
  stationId,
  variableKey,
  mode,
  onSelection,
}: {
  stationId: string;
  variableKey: string;
  mode: ViewMode;
  onSelection: (station: string, variable: string, mode: ViewMode) => void;
}) {
  const { t, locale } = useI18n();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const selected =
    stations.find((s) => s.id === stationId) ??
    stations.find((s) => s.hasExample)!;
  const current =
    selected.variables.find((v) => v.key === variableKey) ??
    selected.variables[0];
  const view =
    mode === "example" && !current?.example && current?.history
      ? "history"
      : mode === "history" && !current?.history && current?.example
        ? "example"
        : mode;
  const filters: {
    key: Filter;
    label: string;
    matches: (s: Station) => boolean;
  }[] = [
    {
      key: "all",
      label: t("map.filter.all", { n: catalogCounts.stations }),
      matches: () => true,
    },
    {
      key: "history",
      label: t("map.filter.history", { n: catalogCounts.stationsWithHistory }),
      matches: (s) => s.hasHistory,
    },
    {
      key: "validation",
      label: t("map.filter.validation", {
        n: catalogCounts.stationsWithValidation,
      }),
      matches: (s) => s.hasValidation,
    },
    {
      key: "examples",
      label: t("map.filter.examples", {
        n: catalogCounts.stationsWithExamples,
      }),
      matches: (s) => s.hasExample,
    },
  ];
  const normalize = (value: string) =>
    value
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase();
  const list = stations.filter(
    (station) =>
      filters.find((f) => f.key === filter)!.matches(station) &&
      normalize(`${station.id} ${station.name}`).includes(
        normalize(search.trim()),
      ),
  );
  const chooseStation = (station: Station) => {
    const variable =
      station.variables.find((v) => v.key === current?.key) ??
      station.variables[0];
    onSelection(station.id, variable?.key ?? "", mode);
  };
  const date = (value: string | null) => value?.slice(0, 10) ?? "—";
  const metrics = current?.metrics;
  const unit = current ? (unitLabels[current.unit] ?? current.unit) : "";
  return (
    <div className="space-y-8">
      <div className="max-w-3xl">
        <h1 className="font-display text-4xl font-bold sm:text-5xl">
          {t("map.title")}
        </h1>
        <p className="mt-3 text-muted-foreground">{t("copy.description")}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {filters.map((item) => (
          <button
            key={item.key}
            onClick={() => setFilter(item.key)}
            aria-pressed={filter === item.key}
            className={`rounded-full px-4 py-2 text-sm ${filter === item.key ? "bg-accent text-accent-foreground" : "glass hover:bg-foreground/10"}`}
          >
            {item.label}
          </button>
        ))}
        <label className="glass flex items-center gap-2 rounded-full px-4 py-2 text-sm">
          <Search className="h-4 w-4" />
          <span className="sr-only">{t("map.search")}</span>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={t("map.search")}
            className="w-40 bg-transparent placeholder:text-muted-foreground"
          />
        </label>
      </div>
      <p className="text-sm text-muted-foreground">
        {t("map.overlapNote", { n: catalogCounts.catalogOnly })}
      </p>
      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <TrentinoMap
            selected={selected}
            onSelect={chooseStation}
            list={list}
          />
        </div>
        <section
          className="glass flex min-h-0 flex-col overflow-hidden rounded-3xl lg:col-span-2 lg:h-0 lg:min-h-full"
          aria-labelledby="station-list-title"
        >
          <div className="shrink-0 border-b border-foreground/10 px-5 py-4">
            <h2
              id="station-list-title"
              className="font-display text-lg font-semibold"
            >
              {t("map.listTitle")}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {t("map.listHint", { n: list.length })}
            </p>
          </div>
          <ul className="min-h-0 max-h-72 flex-1 divide-y divide-foreground/10 overflow-y-auto [scrollbar-gutter:stable] lg:max-h-none">
            {list.map((station) => (
              <li key={station.id}>
                <button
                  onClick={() => chooseStation(station)}
                  aria-pressed={selected.id === station.id}
                  aria-controls="station-details"
                  className={`flex w-full items-center gap-3 border-l-4 px-4 py-3 text-left transition-colors focus-visible:-outline-offset-4 ${selected.id === station.id ? "border-accent bg-accent/10" : "border-transparent hover:bg-foreground/10"}`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold">
                      {station.name}
                    </span>
                    <span className="mt-1 block text-xs text-muted-foreground">
                      {station.id} ·{" "}
                      {t(
                        station.hasHistory
                          ? "map.tag.history"
                          : "map.tag.catalog",
                      )}
                    </span>
                  </span>
                  {selected.id === station.id ? (
                    <Check
                      aria-hidden="true"
                      className="h-5 w-5 shrink-0 text-accent"
                    />
                  ) : (
                    <ChevronRight
                      aria-hidden="true"
                      className="h-5 w-5 shrink-0 text-muted-foreground"
                    />
                  )}
                </button>
              </li>
            ))}
            {!list.length && (
              <li className="p-4 text-center">{t("map.empty")}</li>
            )}
          </ul>
        </section>
      </div>
      <section
        id="station-details"
        className="glass scroll-mt-28 rounded-3xl p-5 sm:p-8"
        aria-labelledby="station-title"
      >
        <div className="grid gap-6 border-b border-foreground/10 pb-6 md:grid-cols-2">
          <div>
            <p className="text-xs uppercase tracking-widest text-accent">
              {t("map.selectedStation")} · {selected.id}
            </p>
            <h2
              id="station-title"
              className="mt-1 font-display text-2xl font-bold"
              aria-live="polite"
            >
              {selected.name}
            </h2>
            <p className="mt-2 flex items-center gap-1 text-sm text-muted-foreground">
              <MapPin aria-hidden="true" className="h-4 w-4" />
              {selected.lat.toFixed(4)}°N, {selected.lon.toFixed(4)}°E
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              {selected.elevation === null
                ? t("map.noElevation")
                : t("map.elevation", {
                    v: selected.elevation.toLocaleString(locale),
                  })}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {selected.startDate
                ? t("map.from", { v: date(selected.startDate) })
                : ""}
              {selected.endDate
                ? t("map.to", { v: date(selected.endDate) })
                : t("map.active")}
            </p>
            <p className="mt-4 text-xs text-muted-foreground">
              {t("map.sourceLabel", { v: attribution })}
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold">{t("map.availability")}</h3>
            <dl className="mt-2 divide-y divide-foreground/10">
              {[
                [t("map.badge.history"), selected.hasHistory],
                [t("map.badge.validation"), selected.hasValidation],
                [t("map.badge.examples"), selected.hasExample],
              ].map(([label, available]) => (
                <div
                  key={String(label)}
                  className="flex flex-wrap justify-between gap-x-4 gap-y-1 py-2 text-sm"
                >
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd
                    className={
                      available
                        ? "font-medium text-foreground"
                        : "text-muted-foreground"
                    }
                  >
                    {t(available ? "map.available" : "map.unavailable")}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
        {!current && (
          <p className="mt-4 text-sm text-muted-foreground">
            {t("map.catalogOnly")}
          </p>
        )}
        {current && (
          <div className="pt-6">
            <fieldset className="mb-5">
              <legend className="mb-3 text-sm font-semibold">
                {t("map.variableLabel")}
              </legend>
              <div className="flex flex-wrap gap-2">
                {selected.variables.map((variable) => (
                  <button
                    key={variable.key}
                    aria-pressed={current.key === variable.key}
                    aria-controls="station-chart"
                    onClick={() => onSelection(selected.id, variable.key, view)}
                    className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${current.key === variable.key ? "border-accent bg-accent text-accent-foreground" : "border-foreground/25 hover:border-foreground/50 hover:bg-foreground/10"}`}
                  >
                    {t(`var.${variable.key}`)}
                  </button>
                ))}
              </div>
            </fieldset>
            <h3 className="font-display text-xl font-semibold sm:text-2xl">
              {t(`var.${current.key}`)}{" "}
              <span className="text-base font-normal text-muted-foreground">
                ({unit})
              </span>
            </h3>
            <div className="my-5 flex flex-wrap gap-3">
              {(["example", "history"] as const).map((item) => (
                <button
                  key={item}
                  aria-pressed={view === item}
                  disabled={
                    item === "example" ? !current.example : !current.history
                  }
                  onClick={() => onSelection(selected.id, current.key, item)}
                  aria-controls="station-chart"
                  className={`border-b-2 px-1 py-2 text-sm disabled:opacity-40 ${view === item ? "border-accent text-accent" : "border-transparent text-muted-foreground enabled:hover:text-foreground"}`}
                >
                  {t(`chart.${item}`)}
                </button>
              ))}
            </div>
            <div id="station-chart">
              {view === "example" ? (
                <ReconstructionExample
                  key={`${selected.id}/${current.key}`}
                  variable={current}
                />
              ) : (
                <HistoricalSeries
                  key={`${selected.id}/${current.key}`}
                  variable={current}
                />
              )}
            </div>
          </div>
        )}
      </section>
      {current && (
        <section className="glass rounded-3xl p-6">
          <h2 className="text-lg font-semibold">
            {t("map.validationTitle")} · {t(`var.${current.key}`)}
          </h2>
          {metrics ? (
            <>
              <div className="my-4 flex flex-wrap gap-8">
                {[
                  ["MAE", metrics.mae],
                  ["RMSE", metrics.rmse],
                ].map(([label, value]) => (
                  <div key={label}>
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="font-display text-2xl text-accent">
                      {Number(value).toLocaleString(locale, {
                        maximumFractionDigits: 3,
                      })}{" "}
                      {unit}
                    </p>
                  </div>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                {t("map.resultNote", {
                  groups: metrics.groups,
                  expected: metrics.expectedGroups,
                })}
              </p>
            </>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">
              {t("copy.noValidation")}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
