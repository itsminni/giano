import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useI18n } from "./i18n";
import { ZoomView } from "./ZoomView";
import { type StationVariable, unitLabels } from "./data";
import {
  assetUrl,
  examplePoints,
  hiddenRanges,
  historyPoints,
  loadAsset,
  shiftDay,
  timeLabel,
  wallTime,
  type Example,
  type HistoryChunk,
  type HistoryIndex,
  type SeriesPoint,
} from "./series";

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--glass-border)",
  borderRadius: 12,
  color: "var(--foreground)",
};

function DataState({ error, retry }: { error: boolean; retry: () => void }) {
  const { t } = useI18n();
  return (
    <div role="status" className="glass rounded-2xl p-6">
      <p>{t(error ? "data.error" : "data.loading")}</p>
      {error && (
        <button className="mt-3 text-accent underline" onClick={retry}>
          {t("data.retry")}
        </button>
      )}
    </div>
  );
}

function SeriesChart({
  points,
  unit,
  direction,
  showTruth = false,
  ranges = [],
}: {
  points: SeriesPoint[];
  unit: string;
  direction: boolean;
  showTruth?: boolean;
  ranges?: { start: string; end: string }[];
}) {
  const { t, locale } = useI18n();
  // Wind direction uses points to avoid lines across the 0°/360° boundary.
  const [showReconstruction, setShowReconstruction] = useState(true);
  const [showEra5, setShowEra5] = useState(false);
  const hasEra5 = points.some((point) => point.era5 != null);
  const strokeWidth = direction ? 0 : 1.8;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-x-6 gap-y-1">
        <label className="flex min-h-11 w-fit cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showReconstruction}
            onChange={(event) => setShowReconstruction(event.target.checked)}
            className="h-4 w-4 accent-[var(--accent)]"
          />
          {t("chart.showReconstruction")}
        </label>
        {hasEra5 && (
          <label className="flex min-h-11 w-fit cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={showEra5}
              onChange={(event) => setShowEra5(event.target.checked)}
              className="h-4 w-4 accent-[var(--accent)]"
            />
            {t("chart.showEra5")}
          </label>
        )}
      </div>
      {hasEra5 && showEra5 && (
        <p className="text-xs text-muted-foreground">{t("chart.era5Note")}</p>
      )}
      <ZoomView
        key={`${points[0]?.time}-${points.at(-1)?.time}`}
        label={`${t("history.values")} · ${unit}`}
        length={points.length}
      >
        {(start, end) => (
          <div
            className="h-[320px] w-full sm:h-[400px]"
            role="img"
            aria-label={`${t("history.values")} · ${unit}`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={points.slice(start, end)}
                margin={{ top: 12, right: 12, left: 0, bottom: 12 }}
                accessibilityLayer
              >
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="time"
                  minTickGap={65}
                  tickFormatter={(value: string) => timeLabel(value).slice(5)}
                  stroke="var(--muted-foreground)"
                  fontSize={11}
                />
                <YAxis
                  includeHidden
                  width={64}
                  domain={direction ? [0, 360] : ["auto", "auto"]}
                  stroke="var(--muted-foreground)"
                  fontSize={11}
                  tickFormatter={(value: number) =>
                    value.toLocaleString(locale, { maximumFractionDigits: 1 })
                  }
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(value) => timeLabel(String(value))}
                  formatter={(value: number) =>
                    `${value.toLocaleString(locale, { maximumFractionDigits: 3 })} ${unit}`
                  }
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 14 }} />
                {ranges
                  .filter(
                    (range) =>
                      range.end >= points[start]!.time &&
                      range.start <=
                        points[Math.min(end, points.length) - 1]!.time,
                  )
                  .map((range) => (
                    <ReferenceArea
                      key={range.start}
                      x1={
                        range.start < points[start]!.time
                          ? points[start]!.time
                          : range.start
                      }
                      x2={
                        range.end >
                        points[Math.min(end, points.length) - 1]!.time
                          ? points[Math.min(end, points.length) - 1]!.time
                          : range.end
                      }
                      fill="var(--accent)"
                      fillOpacity={0.1}
                    />
                  ))}
                <Line
                  type="linear"
                  dataKey="observed"
                  name={t("chart.visible")}
                  stroke="var(--chart-1)"
                  strokeWidth={strokeWidth}
                  dot={{ r: direction ? 2 : 1 }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
                {showTruth && (
                  <Line
                    type="linear"
                    dataKey="hidden"
                    name={t("chart.hidden")}
                    stroke="var(--foreground)"
                    strokeWidth={strokeWidth}
                    strokeDasharray="2 4"
                    dot={{ r: 3 }}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                )}
                <Line
                  type="linear"
                  dataKey="reconstructed"
                  hide={!showReconstruction}
                  name={t("chart.reconstruction")}
                  stroke="var(--accent)"
                  strokeWidth={strokeWidth}
                  strokeDasharray="6 3"
                  dot={{ r: direction ? 2 : 1.5 }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
                {hasEra5 && showEra5 && (
                  <Line
                    type="linear"
                    dataKey="era5"
                    name="ERA5-Land"
                    stroke="#c4a5ff"
                    strokeWidth={strokeWidth}
                    strokeDasharray="3 5"
                    dot={direction ? { r: 2 } : false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </ZoomView>
    </div>
  );
}

export function ReconstructionExample({
  variable,
}: {
  variable: StationVariable;
}) {
  const { t, locale } = useI18n();
  const [showTruth, setShowTruth] = useState(true);
  const path = variable.example?.json;
  const query = useQuery({
    queryKey: ["example", path],
    enabled: !!path,
    queryFn: ({ signal }) => loadAsset<Example>(path!, signal),
    staleTime: Infinity,
  });
  if (!path) return <p>{t("chart.noExample")}</p>;
  if (!query.data)
    return (
      <DataState
        error={query.isError}
        retry={() => {
          void query.refetch();
        }}
      />
    );
  const example = query.data;
  const unit = unitLabels[example.unit] ?? example.unit;
  const format = (value: number) =>
    `${value.toLocaleString(locale, { maximumSignificantDigits: 3 })} ${unit}`;
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">{t("chart.exampleNote")}</p>
      <div className="flex flex-wrap justify-between gap-3 text-sm">
        <p>
          {timeLabel(example.series.timestamps[0]!)} —{" "}
          {timeLabel(example.series.timestamps.at(-1)!)}
        </p>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showTruth}
            onChange={(event) => setShowTruth(event.target.checked)}
          />
          {t("chart.showTruth")}
        </label>
      </div>
      <SeriesChart
        points={examplePoints(example)}
        unit={unit}
        direction={variable.key === "wind_direction"}
        showTruth={showTruth}
        ranges={hiddenRanges(example)}
      />
      <p className="font-medium">
        {t("chart.exampleStats", {
          hours: example.series.timestamps.length,
          hidden: example.metrics.hidden_points,
          mae: format(example.metrics.mae),
          rmse: format(example.metrics.rmse),
        })}
      </p>
      <a
        href={assetUrl(path)}
        download
        className="inline-block text-sm text-accent underline"
      >
        {t("data.download")}
      </a>
    </div>
  );
}

export function HistoricalSeries({ variable }: { variable: StationVariable }) {
  const { t } = useI18n();
  const path = variable.history;
  const query = useQuery({
    queryKey: ["history", path],
    enabled: !!path,
    queryFn: ({ signal }) => loadAsset<HistoryIndex>(path!, signal),
    staleTime: Infinity,
  });
  if (!path) return <p>{t("chart.noHistory")}</p>;
  if (!query.data)
    return (
      <DataState
        error={query.isError}
        retry={() => {
          void query.refetch();
        }}
      />
    );
  return <HistoryDetail key={path} index={query.data} />;
}

function HistoryDetail({ index }: { index: HistoryIndex }) {
  const { t, locale } = useI18n();
  const [start, setStart] = useState(index.start.slice(0, 10));
  const [days, setDays] = useState(7);
  const lower = wallTime(start),
    upper = lower + days * 86400000;
  const paths = index.chunks
    .filter(
      (chunk) => wallTime(chunk.start) < upper && wallTime(chunk.end) >= lower,
    )
    .map((chunk) => chunk.path);
  const query = useQuery({
    queryKey: ["history-chunks", paths],
    queryFn: ({ signal }) =>
      Promise.all(paths.map((path) => loadAsset<HistoryChunk>(path, signal))),
    staleTime: Infinity,
  });
  const points = query.data ? historyPoints(query.data, start, days) : [];
  const clampDate = (date: string) =>
    date < index.start.slice(0, 10)
      ? index.start.slice(0, 10)
      : date > index.end.slice(0, 10)
        ? index.end.slice(0, 10)
        : date;
  const unit = unitLabels[index.unit] ?? index.unit;
  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">{t("history.note")}</p>
      <div className="grid gap-3 sm:grid-cols-3">
        {[
          ["history.observed", index.observed_hours],
          ["history.reconstructed", index.reconstructed_hours],
          ["history.unfilled", index.unfilled_hours],
        ].map(([label, value]) => (
          <div key={label} className="glass rounded-2xl p-4">
            <p className="text-sm text-muted-foreground">{t(String(label))}</p>
            <p className="text-xl font-semibold">
              {Number(value).toLocaleString(locale)}
            </p>
          </div>
        ))}
      </div>
      <div>
        <h4 className="font-semibold">{t("history.overview")}</h4>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("history.overviewNote")}
        </p>
        <ZoomView
          label={t("history.overview")}
          length={index.monthly_overview.length}
        >
          {(start, end) => (
            <div className="mt-4 h-44">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={index.monthly_overview.slice(start, end)}
                  accessibilityLayer
                >
                  <XAxis
                    dataKey="month"
                    minTickGap={70}
                    stroke="var(--muted-foreground)"
                    fontSize={10}
                  />
                  <YAxis
                    width={42}
                    stroke="var(--muted-foreground)"
                    fontSize={10}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar
                    dataKey="observed_hours"
                    name={t("history.observed")}
                    stackId="hours"
                    fill="var(--chart-1)"
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="reconstructed_hours"
                    name={t("history.reconstructed")}
                    stackId="hours"
                    fill="var(--accent)"
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="unfilled_hours"
                    name={t("history.unfilled")}
                    stackId="hours"
                    fill="var(--muted-foreground)"
                    isAnimationActive={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </ZoomView>
      </div>
      <div className="flex flex-wrap items-end gap-4 text-sm">
        <label className="grid gap-1">
          {t("history.month")}
          <select
            value={start.slice(0, 7)}
            onChange={(event) =>
              setStart(clampDate(`${event.target.value}-01`))
            }
            className="rounded-xl border border-border bg-background p-2"
          >
            {index.monthly_overview.map((month) => (
              <option key={month.month} value={month.month}>
                {month.month}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1">
          {t("history.start")}
          <input
            type="date"
            value={start}
            min={index.start.slice(0, 10)}
            max={index.end.slice(0, 10)}
            onChange={(event) => {
              if (event.target.value) setStart(clampDate(event.target.value));
            }}
            className="rounded-xl border border-border bg-background p-2"
          />
        </label>
        <label className="grid gap-1">
          {t("history.duration")}
          <select
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
            className="rounded-xl border border-border bg-background p-2"
          >
            {[3, 7, 30].map((value) => (
              <option key={value} value={value}>
                {t("history.days", { n: value })}
              </option>
            ))}
          </select>
        </label>
        <button
          className="glass rounded-full px-4 py-2 disabled:opacity-40"
          disabled={start <= index.start.slice(0, 10)}
          onClick={() => setStart(clampDate(shiftDay(start, -days)))}
        >
          {t("history.previous")}
        </button>
        <button
          className="glass rounded-full px-4 py-2 disabled:opacity-40"
          disabled={upper > wallTime(index.end)}
          onClick={() => setStart(clampDate(shiftDay(start, days)))}
        >
          {t("history.next")}
        </button>
      </div>
      {!query.data ? (
        <DataState
          error={query.isError}
          retry={() => {
            void query.refetch();
          }}
        />
      ) : points.length ? (
        <SeriesChart
          points={points}
          unit={unit}
          direction={index.variable === "wind_direction"}
        />
      ) : (
        <p>{t("history.empty")}</p>
      )}
      <p className="text-xs text-muted-foreground">
        {t("history.fullRange", {
          start: timeLabel(index.start),
          end: timeLabel(index.end),
        })}
      </p>
      <div className="flex flex-wrap gap-4 text-sm">
        {paths.map((path) => (
          <a
            key={path}
            href={assetUrl(path)}
            download
            className="text-accent underline"
          >
            {t("data.download")}
            {import.meta.env.MODE === "pages" ? " (.gz)" : ""} ·{" "}
            {path.split("/").at(-1)?.replace(".json", "")}
          </a>
        ))}
      </div>
    </div>
  );
}
