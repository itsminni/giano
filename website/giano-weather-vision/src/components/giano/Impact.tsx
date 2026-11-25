import { ArrowRight } from "lucide-react";
import { extendedRepairResults, GITHUB_URL } from "./data";
import { useI18n } from "./i18n";
import type { TabId } from "./Nav";

export function Impact({ onNavigate }: { onNavigate?: (tab: TabId) => void }) {
  const { t, locale } = useI18n();

  return (
    <div className="space-y-16">
      <div className="max-w-2xl">
        <h2 className="font-display text-4xl font-bold sm:text-5xl">
          {t("impact.title")}
        </h2>
        <p className="mt-3 text-muted-foreground">{t("impact.intro")}</p>
      </div>

      <section className="grid gap-6 md:grid-cols-2">
        {(["swiss", "germany"] as const).map((study) => (
          <article key={study} className="glass rounded-[2rem] p-6 sm:p-8">
            <h3 className="font-display text-2xl font-bold">
              {t(`impact.${study}Title`)}
            </h3>
            <p className="mt-3 max-w-prose text-muted-foreground">
              {t(`impact.${study}Design`)}
            </p>
            <p className="mt-5 font-display text-3xl font-bold text-accent">
              {t(`impact.${study}Score`)}
            </p>
            <a
              className="mt-4 inline-block text-sm text-accent underline"
              href={`${GITHUB_URL}/tree/main/examples/${study === "swiss" ? "swiss" : "dwd"}`}
            >
              {t("impact.studyLink")}
            </a>
          </article>
        ))}
      </section>
      <section className="glass rounded-[2rem] p-6 sm:p-8">
        <h3 className="mt-2 font-display text-3xl font-bold">
          {t("impact.downstreamTitle")}
        </h3>
        <p className="mt-4 max-w-prose leading-relaxed text-muted-foreground">
          {t("impact.downstreamIntro")}
        </p>
        <div
          className="mt-6 max-w-full overflow-x-auto"
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
                  {t("impact.linear")}
                </th>
                <th scope="col" className="py-3 pl-3 text-right">
                  {t("impact.stationMode")}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-foreground/10">
              {extendedRepairResults.map((row) => {
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
                      {format(row.linear)}
                    </td>
                    <td className="py-4 pl-3 text-right tabular-nums">
                      {format(row.giano)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <a
          className="mt-4 inline-block text-sm text-accent underline"
          href={`${GITHUB_URL}/wiki/Downstream-Utility`}
        >
          {t("impact.downstreamLink")}
        </a>
      </section>

      <section className="glass-strong rounded-[2rem] p-6 sm:p-8">
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
          <a
            href="#guida"
            onClick={
              onNavigate
                ? (event) => {
                    event.preventDefault();
                    onNavigate("guida");
                  }
                : undefined
            }
            className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-semibold text-accent-foreground"
          >
            {t("impact.useGuide")} <ArrowRight className="h-4 w-4" />
          </a>
          <a className="text-accent underline" href={GITHUB_URL}>
            {t("impact.useCode")}
          </a>
        </div>
      </section>
    </div>
  );
}
