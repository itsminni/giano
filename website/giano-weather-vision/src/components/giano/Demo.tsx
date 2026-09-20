import { useEffect, useState } from "react";
import { Github, Pause, Play, RotateCcw } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { demoSeries, GITHUB_URL } from "./data";
import { useI18n } from "./i18n";
import { ZoomView } from "./ZoomView";

const STEP_KEYS = ["demo.step1", "demo.step2", "demo.step3", "demo.step4", "demo.step5"];

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--glass-border)",
  borderRadius: "12px",
  color: "var(--popover-foreground)",
  fontSize: 12,
};

export function Demo() {
  const { t } = useI18n();
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const id = setTimeout(() => setStep((s) => (s + 1) % STEP_KEYS.length), step === 1 ? 4000 : 1800);
    return () => clearTimeout(id);
  }, [playing, step]);

  const showGap = step >= 2;
  const data = demoSeries.map((d) => ({
    ...d,
    giano: step >= 3 && d.t >= 15 && d.t <= 31 ? d.vero : null,
  }));

  return (
    <div className="space-y-10">
      <div className="max-w-2xl">
        <h2 className="font-display text-4xl font-bold sm:text-5xl">{t("demo.title")}</h2>
        <p className="mt-3 text-muted-foreground">{t("demo.intro")}</p>
      </div>

      <div className="glass-strong rounded-[2rem] p-6 sm:p-8">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-2.5 text-sm font-semibold text-accent-foreground"
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {playing ? t("demo.pause") : t("demo.play")}
          </button>
          <button
            onClick={() => {
              setStep(0);
              setPlaying(true);
            }}
            className="glass inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm"
          >
            <RotateCcw className="h-4 w-4" /> {t("demo.restart")}
          </button>
          <p className="ml-auto text-sm text-muted-foreground">
            {t("demo.step", {
              i: step + 1,
              n: STEP_KEYS.length,
              label: t(STEP_KEYS[step] ?? STEP_KEYS[0]!),
            })}
          </p>
        </div>

        <div className="mt-5 flex gap-1.5">
          {STEP_KEYS.map((s, i) => (
            <button
              key={s}
              onClick={() => {
                setStep(i);
                setPlaying(false);
              }}
              className={`h-1.5 flex-1 rounded-full transition-colors ${
                i <= step ? "bg-accent" : "bg-foreground/15"
              }`}
              aria-label={t(s)}
            />
          ))}
        </div>

        <div className="mt-8">
          <ZoomView label={t("demo.title")} length={data.length}>
          {(start, end) => (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data.slice(start, end)} margin={{ top: 12, right: 12, bottom: 4, left: 0 }} accessibilityLayer>
              <CartesianGrid stroke="var(--border)" vertical={false} />
              <XAxis dataKey="t" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis
                domain={[5, 15]}
                ticks={[5, 7.5, 10, 12.5, 15]}
                stroke="var(--muted-foreground)"
                fontSize={11}
                unit="°"
              />
              <Tooltip contentStyle={tooltipStyle} />
              {step >= 1 && (
                <Line
                  type="monotone"
                  dataKey="era5"
                  name={t("demo.era5")}
                  stroke="var(--chart-5)"
                  strokeWidth={step === 1 ? 3 : 1.5}
                  strokeOpacity={step === 1 ? 1 : 0.45}
                  strokeDasharray="3 5"
                  dot={step === 1 ? { r: 2 } : false}
                  isAnimationActive={false}
                />
              )}
              {step === 1 && (
                <ReferenceLine
                  x={24}
                  stroke="var(--chart-5)"
                  strokeDasharray="4 4"
                  label={{ value: t("demo.alignedHour"), fill: "var(--chart-5)", fontSize: 12, position: "insideTopRight" }}
                />
              )}
              {showGap && start <= 30 && end > 16 && (
                <ReferenceArea
                  x1={Math.max(16, start)}
                  x2={Math.min(30, end - 1)}
                  fill="var(--chart-2)"
                  fillOpacity={0.22}
                  stroke="var(--chart-2)"
                  strokeOpacity={0.55}
                  label={{
                    value: "GAP",
                    fill: "var(--chart-2)",
                    fontSize: 12,
                    position: "insideTop",
                  }}
                />
              )}
              <Line
                type="monotone"
                dataKey="osservato"
                name={t("demo.observed")}
                stroke="var(--chart-1)"
                strokeWidth={2.5}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
              {step >= 3 && (
                <Line
                  key={step}
                  type="monotone"
                  dataKey="giano"
                  name={t("demo.reconstructed")}
                  stroke="var(--chart-2)"
                  strokeWidth={3}
                  strokeDasharray="7 5"
                  dot={false}
                  connectNulls
                  isAnimationActive
                  animationDuration={step === 3 ? 1500 : 450}
                  animationEasing="linear"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
          )}
          </ZoomView>
        </div>

        <div className="mt-4 min-h-20 text-sm" aria-live="polite">
          {step >= 1 && (
            <p className="flex flex-wrap gap-x-5 gap-y-2">
              <span className="text-[var(--chart-1)]">— {t("demo.observed")}</span>
              <span className="text-[var(--chart-5)]">··· {t("demo.era5")}</span>
              {step >= 3 && <span className="text-[var(--chart-2)]">– – {t("demo.reconstructed")}</span>}
            </p>
          )}
          {step === 1 && <p className="mt-2 text-muted-foreground">{t("demo.alignmentNote")}</p>}
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-5">
          {STEP_KEYS.map((s, i) => (
            <div
              key={s}
              className={`glass rounded-2xl p-3 text-xs transition-opacity ${
                i <= step ? "opacity-100" : "opacity-40"
              }`}
            >
              <span className={i <= step ? "text-accent" : "text-muted-foreground"}>0{i + 1}</span>
              <p className="mt-1 text-muted-foreground">{t(s)}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="glass flex flex-col items-start gap-4 rounded-[2rem] p-8 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="font-display text-2xl font-bold">{t("demo.openTitle")}</h3>
          <p className="mt-1 text-muted-foreground">{t("demo.openText")}</p>
        </div>
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex shrink-0 items-center gap-2 rounded-full bg-accent px-6 py-3 font-semibold text-accent-foreground transition-transform hover:scale-[1.03]"
        >
          <Github className="h-5 w-5" /> {t("demo.repo")}
        </a>
      </div>
    </div>
  );
}
