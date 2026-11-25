import { useState } from "react";
import { useHydrated } from "@tanstack/react-router";
import { Copy, ExternalLink } from "lucide-react";
import { useI18n } from "./i18n";
import { DATASET_URL, GITHUB_URL } from "./data";

const install = `git clone https://github.com/itsminni/giano.git
cd giano
uv sync --locked --python 3.11
uv run --locked pytest -q`;
const reconstruct = `uv run --locked giano --input weather.nc \\
  --start 2024-03-01 --end-exclusive 2024-04-01 \\
  --variables temperature --local-context --station-reference --output march.nc`;
const importData = `uv run --locked giano --input examples/noaa-2024/observations.csv --stations examples/noaa-2024/stations.csv \\
  --start 2024-03-01 --end-exclusive 2024-04-01 --local-context --station-reference --output swiss-march.nc`;

function Command({ value }: { value: string }) {
  const { t } = useI18n();
  const [status, setStatus] = useState("guide.copy");
  const hydrated = useHydrated();
  return (
    <div className="my-5 overflow-hidden rounded-2xl border border-border bg-background/70">
      {hydrated && (
        <div className="flex justify-end border-b border-border px-4 py-2">
          <button
            className="flex items-center gap-2 text-xs text-accent"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(value);
                setStatus("guide.copied");
              } catch {
                setStatus("guide.copyFailed");
              }
            }}
          >
            <Copy className="h-3.5 w-3.5" />
            <span aria-live="polite">{t(status)}</span>
          </button>
        </div>
      )}
      <pre className="overflow-x-auto p-5 text-sm leading-7">
        <code>{value}</code>
      </pre>
    </div>
  );
}
export function Guide() {
  const { t } = useI18n();
  return (
    <div className="space-y-8">
      <header>
        <h1 className="font-display text-4xl font-bold sm:text-5xl">
          {t("guide.title")}
        </h1>
        <p className="mt-4 text-lg text-muted-foreground">{t("guide.intro")}</p>
      </header>
      <section className="glass rounded-3xl p-6 sm:p-8">
        <h2 className="text-2xl font-semibold">{t("guide.installTitle")}</h2>
        <p className="mt-3 text-muted-foreground">{t("guide.installText")}</p>
        <Command value={install} />
      </section>
      <section className="glass rounded-3xl p-6 sm:p-8">
        <h2 className="text-2xl font-semibold">{t("guide.weightsTitle")}</h2>
        <p className="mt-3 text-muted-foreground">{t("guide.weightsText")}</p>
        <a
          className="mt-4 mr-6 inline-flex items-center gap-2 text-accent underline"
          href={DATASET_URL}
        >
          {t("guide.downloadData")} <ExternalLink className="h-4 w-4" />
        </a>
        <a
          className="mt-4 inline-flex items-center gap-2 text-accent underline"
          href={`${GITHUB_URL}/wiki/Input-and-output`}
        >
          {t("guide.release")} <ExternalLink className="h-4 w-4" />
        </a>
      </section>
      <section className="glass rounded-3xl p-6 sm:p-8">
        <h2 className="text-2xl font-semibold">{t("guide.runTitle")}</h2>
        <p className="mt-3 text-muted-foreground">{t("guide.runText")}</p>
        <Command value={reconstruct} />
        <p className="text-sm text-muted-foreground">
          {t("guide.commandNote")}
        </p>
        <p className="mt-3 max-w-prose text-sm text-muted-foreground">
          {t("guide.stationOptions")}
        </p>
      </section>
      <section className="glass rounded-3xl p-6 sm:p-8">
        <h2 className="text-2xl font-semibold">{t("guide.datasetTitle")}</h2>
        <p className="mt-3 text-muted-foreground">{t("guide.datasetText")}</p>
        <Command value={importData} />
        <p className="text-sm text-muted-foreground">
          {t("guide.exampleText")}
        </p>
        <a
          className="mt-4 inline-block text-accent underline"
          href={`${GITHUB_URL}/wiki/Input-and-output`}
        >
          {t("guide.dataset")}
        </a>
      </section>
      <section className="glass rounded-3xl p-6 sm:p-8">
        <h2 className="text-2xl font-semibold">{t("guide.trainTitle")}</h2>
        <p className="mt-3 text-muted-foreground">{t("guide.trainText")}</p>
        <Command
          value={`uv run --locked giano-prepare --data data --output runs/windows
uv run --locked giano-train --cache runs/windows --output runs/training --fixed-rounds
uv run --locked giano-evaluate --data data --ensemble runs/training/ensemble.pt --local-context --station-reference --output runs/evaluation`}
        />
        <a
          className="text-accent underline"
          href={`${GITHUB_URL}/wiki/Training-and-reproduction`}
        >
          {t("guide.trainLink")}
        </a>
      </section>
      <a
        className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-semibold text-accent-foreground"
        href={`${GITHUB_URL}/wiki`}
      >
        {t("guide.wiki")} <ExternalLink className="h-4 w-4" />
      </a>
    </div>
  );
}
