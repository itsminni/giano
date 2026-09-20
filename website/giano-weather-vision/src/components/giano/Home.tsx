import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  Activity,
  Database,
  Satellite,
  Target,
} from "lucide-react";
import heroImg from "@/assets/hero-trentino.jpg";
import {
  attribution,
  catalogCounts,
  validationInfo,
  validationMetrics,
} from "./data";
import { useI18n } from "./i18n";
import { TrentinoMap } from "./TrentinoMap";
import type { TabId } from "./Nav";

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--glass-border)",
  borderRadius: "12px",
  backdropFilter: "blur(16px)",
  color: "var(--popover-foreground)",
  fontSize: 12,
};

function Stat({
  icon: Icon,
  value,
  label,
}: {
  icon: typeof Target;
  value: string;
  label: string;
}) {
  return (
    <div className="glass rounded-3xl p-5">
      <Icon className="h-5 w-5 text-accent" />
      <p className="mt-3 font-display text-3xl font-bold">{value}</p>
      <p className="text-sm text-muted-foreground">{label}</p>
    </div>
  );
}

export function Home({ onNavigate }: { onNavigate: (t: TabId) => void }) {
  const { t, locale } = useI18n();
  return (
    <div className="space-y-16">
      <section className="glass relative overflow-hidden rounded-[2rem] px-6 py-16 sm:px-12 sm:py-24">
        <img
          src={heroImg}
          alt={t("home.heroAlt")}
          width={1920}
          height={1080}
          className="absolute inset-0 h-full w-full object-cover opacity-40"
        />
        <div className="absolute inset-0 bg-background/40" />
        <div className="relative max-w-2xl">
          <h1 className="font-display text-6xl font-bold sm:text-8xl">
            <span className="text-gradient">Giano</span>
          </h1>
          <p className="mt-3 font-display text-xl sm:text-2xl">
            {t("home.subtitle")}
          </p>
          <p className="mt-4 text-lg text-muted-foreground sm:text-xl">
            {t("copy.description")}
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate("mappa")}
              className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-semibold text-accent-foreground transition-transform hover:scale-[1.03]"
            >
              {t("home.ctaMap", { n: catalogCounts.stations })}{" "}
              <ArrowRight className="h-4 w-4" />
            </button>
            <button
              onClick={() => onNavigate("demo")}
              className="glass rounded-full px-6 py-3 font-medium transition-colors hover:bg-foreground/10"
            >
              {t("home.ctaDemo")}
            </button>
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          icon={Database}
          value={`${catalogCounts.stationsWithHistory}`}
          label={t("home.stat1")}
        />
        <Stat
          icon={Activity}
          value={`${catalogCounts.series}`}
          label={t("home.stat2")}
        />
        <Stat
          icon={Satellite}
          value={catalogCounts.reconstructedHours.toLocaleString(locale)}
          label={t("home.stat3")}
        />
        <Stat
          icon={Target}
          value={`${catalogCounts.stationsWithValidation}`}
          label={t("home.stat4")}
        />
      </section>

      <section className="grid gap-6 lg:grid-cols-5">
        <div className="glass rounded-3xl p-6 lg:col-span-3">
          <h3 className="text-lg font-semibold">{t("home.errorTitle")}</h3>
          <p className="mb-4 text-sm text-muted-foreground">
            {t("copy.results")}
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {validationMetrics.map((m) => (
              <div key={m.key} className="glass rounded-2xl p-4">
                <p className="text-sm font-medium">{t(`var.${m.key}`)}</p>
                <p className="mt-1 font-display text-2xl font-bold text-accent">
                  {m.mae.toLocaleString(locale, { maximumFractionDigits: 2 })}{" "}
                  {m.unit}
                </p>
                <p className="text-xs text-muted-foreground">
                  MAE · RMSE{" "}
                  {m.rmse.toLocaleString(locale, { maximumFractionDigits: 2 })}{" "}
                  {m.unit}
                </p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            {t("home.errorNote")}
          </p>
        </div>

        <div className="glass rounded-3xl p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold">{t("home.coverTitle")}</h3>
          <p className="mb-4 text-sm text-muted-foreground">
            {t("home.coverNote")}
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              layout="vertical"
              data={[
                { k: t("home.cover.catalog"), n: catalogCounts.stations },
                {
                  k: t("home.cover.history"),
                  n: catalogCounts.stationsWithHistory,
                },
                {
                  k: t("home.cover.examples"),
                  n: catalogCounts.stationsWithExamples,
                },
                {
                  k: t("home.cover.validation"),
                  n: catalogCounts.stationsWithValidation,
                },
              ]}

              margin={{ left: 10 }}
            >
              <CartesianGrid stroke="var(--border)" horizontal={false} />
              <XAxis
                type="number"
                stroke="var(--muted-foreground)"
                fontSize={11}
              />
              <YAxis
                type="category"
                dataKey="k"
                width={100}
                stroke="var(--muted-foreground)"
                fontSize={11}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: "var(--glass)" }}
              />
              <Bar
                dataKey="n"
                name={t("home.cover.series")}
                fill="var(--chart-1)"
                radius={[0, 6, 6, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-5">
        <div className="glass rounded-3xl p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold">{t("home.measuredTitle")}</h3>
          <p className="mt-3 text-sm text-muted-foreground">
            {t("copy.example")}
          </p>
          <ul className="mt-5 divide-y divide-foreground/10 text-sm text-muted-foreground">
            <li className="py-3">
              {t("home.validationSet", { n: validationInfo.caseSeedGroups })}
            </li>
            <li className="py-3">
              {t("home.seeds", { v: validationInfo.seeds.join(", ") })}
            </li>
            <li className="py-3">{t("home.source", { v: attribution })}</li>
          </ul>
        </div>

        <div className="lg:col-span-3">
          <div className="mb-4 flex items-end justify-between gap-4">
            <h3 className="text-lg font-semibold">{t("home.networkTitle")}</h3>
            <button
              onClick={() => onNavigate("mappa")}
              className="text-sm font-medium text-accent hover:underline"
            >
              {t("home.openMap")}
            </button>
          </div>
          <TrentinoMap compact />
        </div>
      </section>
    </div>
  );
}
