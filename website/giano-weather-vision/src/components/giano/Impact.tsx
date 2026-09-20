import {
  AlertTriangle,
  CloudOff,
  Droplets,
  LineChart,
  Snowflake,
  TrendingUp,
  ArrowRight,
} from "lucide-react";
import { catalogCounts, downstreamResults, GITHUB_URL } from "./data";
import { useI18n } from "./i18n";
import type { TabId } from "./Nav";

export function Impact({ onNavigate }: { onNavigate: (tab: TabId) => void }) {
  const { t, locale } = useI18n();

  return (
    <div className="space-y-16">
      <div className="max-w-2xl">
        <h2 className="font-display text-4xl font-bold sm:text-5xl">
          {t("impact.title")}
        </h2>
        <p className="mt-3 text-muted-foreground">{t("impact.intro")}</p>
      </div>

      <section className="grid gap-6 lg:grid-cols-12">
        <div className="glass rounded-[2rem] p-8 lg:col-span-7">
          <span className="text-xs uppercase tracking-widest text-accent">
            01
          </span>
          <h3 className="mt-2 font-display text-3xl font-bold">
            {t("impact.h1")}
          </h3>
          <p className="mt-4 text-muted-foreground">{t("impact.p1")}</p>
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              { icon: TrendingUp, t: "impact.c1t", d: "impact.c1d" },
              { icon: Droplets, t: "impact.c2t", d: "impact.c2d" },
              { icon: Snowflake, t: "impact.c3t", d: "impact.c3d" },
            ].map((c) => (
              <div key={c.t} className="glass rounded-2xl p-4">
                <c.icon className="h-5 w-5 text-accent" />
                <p className="mt-2 font-semibold">{t(c.t)}</p>
                <p className="text-sm text-muted-foreground">{t(c.d)}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-strong flex flex-col justify-between rounded-[2rem] p-8 lg:col-span-5">
          <LineChart className="h-6 w-6 text-accent" />
          <div className="mt-8">
            <p className="font-display text-6xl font-bold text-gradient">
              {catalogCounts.stationsWithHistory}
            </p>
            <p className="mt-2 text-muted-foreground">
              {t("impact.statHistory", {
                total: catalogCounts.stations,
                series: catalogCounts.series,
                hours: catalogCounts.reconstructedHours.toLocaleString(locale),
              })}
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-12">
        <div className="glass-strong order-2 rounded-[2rem] p-8 lg:order-1 lg:col-span-4">
          <AlertTriangle className="h-6 w-6 text-accent" />
          <p className="mt-8 font-display text-6xl font-bold text-gradient">
            {catalogCounts.examples}
          </p>
          <p className="mt-2 text-muted-foreground">
            {t("impact.statExamples", {
              n: catalogCounts.stationsWithExamples,
            })}
          </p>
        </div>

        <div className="glass order-1 rounded-[2rem] p-8 lg:order-2 lg:col-span-8">
          <span className="text-xs uppercase tracking-widest text-accent">
            02
          </span>
          <h3 className="mt-2 font-display text-3xl font-bold">
            {t("impact.h2")}
          </h3>
          <p className="mt-4 text-muted-foreground">{t("impact.p2")}</p>
          <ul className="mt-6 space-y-3">
            {[1, 2, 3, 4].map((number) => (
              <li
                key={number}
                className="flex gap-3 text-sm text-muted-foreground"
              >
                <CloudOff className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                <span>
                  <strong className="font-semibold text-foreground">
                    {t(`impact.l${number}Lead`)}
                  </strong>{" "}
                  {t(`impact.l${number}`)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </section>
      <section className="glass rounded-[2rem] p-6 sm:p-8">
        <span className="text-xs uppercase tracking-widest text-accent">
          03 · Downstream
        </span>
        <h3 className="mt-2 font-display text-3xl font-bold">
          {t("impact.downstreamTitle")}
        </h3>
        <p className="mt-4 max-w-4xl leading-relaxed text-muted-foreground">
          {t("impact.downstreamIntro")}
        </p>
        <p className="mt-3 max-w-4xl leading-relaxed text-muted-foreground">
          {t("impact.downstreamProtocol")}
        </p>
        <div
          className="mt-6 overflow-x-auto"
          role="region"
          aria-label={t("impact.downstreamCaption")}
          tabIndex={0}
        >
          <table className="w-full text-left text-sm">
            <caption className="pb-4 text-left font-medium">
              {t("impact.downstreamCaption")}
            </caption>
            <thead className="border-b border-foreground/20 text-muted-foreground">
              <tr>
                <th scope="col" className="py-3 pr-5">
                  {t("impact.downstreamVariable")}
                </th>
                <th scope="col" className="px-3 py-3 text-right">
                  {t("impact.downstreamDamaged")}
                </th>
                <th scope="col" className="px-3 py-3 text-right">
                  {t("impact.downstreamRepaired")}
                </th>
                <th scope="col" className="py-3 pl-3 text-right">
                  {t("impact.downstreamReduction")}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-foreground/10">
              {downstreamResults.map((row) => {
                const reduction =
                  (100 * (row.corrupted - row.giano)) / row.corrupted;
                const format = (value: number) =>
                  value.toLocaleString(locale, {
                    maximumFractionDigits: 3,
                    minimumFractionDigits: 3,
                  });
                return (
                  <tr key={row.variable}>
                    <th scope="row" className="py-4 pr-5 font-medium">
                      {t(`var.${row.variable}`)}{" "}
                      <span className="font-normal text-muted-foreground">
                        ({row.unit})
                      </span>
                      <span className="block text-xs font-normal text-muted-foreground">
                        {row.station}
                      </span>
                    </th>
                    <td className="px-3 py-4 text-right tabular-nums">
                      {format(row.corrupted)}
                    </td>
                    <td className="px-3 py-4 text-right tabular-nums">
                      {format(row.giano)}
                    </td>
                    <td className="py-4 pl-3 text-right tabular-nums">
                      {reduction.toLocaleString(locale, {
                        maximumFractionDigits: 1,
                        minimumFractionDigits: 1,
                      })}
                      %
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-sm text-muted-foreground">
          {t("impact.downstreamBaseline")}
        </p>
        <p className="mt-5 max-w-4xl leading-relaxed text-muted-foreground">
          {t("impact.downstreamConclusion")}
        </p>
        <a
          className="mt-4 inline-block text-sm text-accent underline"
          href={`${GITHUB_URL}/wiki/Downstream-Utility`}
        >
          {t("impact.downstreamLink")}
        </a>
      </section>

      <section className="glass-strong rounded-[2rem] p-6 sm:p-8">
        <span className="text-xs uppercase tracking-widest text-accent">
          04
        </span>
        <h3 className="mt-2 font-display text-3xl font-bold">
          {t("impact.useTitle")}
        </h3>
        <p className="mt-4 max-w-3xl leading-relaxed text-muted-foreground">
          {t("impact.useText")}
        </p>
        <p className="mt-3 max-w-3xl leading-relaxed text-muted-foreground">
          {t("impact.useDetail")}
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-5">
          <button
            onClick={() => onNavigate("guida")}
            className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-semibold text-accent-foreground"
          >
            {t("impact.useGuide")} <ArrowRight className="h-4 w-4" />
          </button>
          <a className="text-accent underline" href={GITHUB_URL}>
            {t("impact.useCode")}
          </a>
        </div>
      </section>
    </div>
  );
}
