import { useState } from "react";
import { Copy, ExternalLink } from "lucide-react";
import { useI18n } from "./i18n";
import { DATASET_URL, GITHUB_URL } from "./data";

const install = `git clone https://github.com/itsminni/giano.git
cd giano
uv sync --locked --python 3.11 --inexact
uv run --locked pytest -q`;
const reconstruct = `uv run --locked giano-impute \\
  --data-dir data/2-processed-v2 --variables temperature \\
  --checkpoint-dir checkpoints/corrected_v2_spectral/imputeformer \\
  --model-family imputeformer --seed 42 --fallback none \\
  --output-dir artifacts/imputed/my-temperature-run`;
const importData = `uv run --locked giano-dataset import \\
  --config /path/to/import.yaml --output-dir data/external/my-network/processed
uv run --locked giano-dataset validate \\
  --data-dir data/external/my-network/processed`;

function Command({ value }: { value: string }) {
  const { t } = useI18n();
  const [status, setStatus] = useState("guide.copy");
  return (
    <div className="my-5 overflow-hidden rounded-2xl border border-border bg-background/70">
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
          href={`${GITHUB_URL}/wiki/Release-and-Reproduction`}
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
          href={`${GITHUB_URL}/wiki/Portability-to-New-Datasets`}
        >
          {t("guide.dataset")}
        </a>
      </section>
      <section className="glass rounded-3xl p-6 sm:p-8">
        <h2 className="text-2xl font-semibold">{t("guide.contextTitle")}</h2>
        <p className="mt-3 text-muted-foreground">{t("guide.contextText")}</p>
        <a
          className="mt-4 inline-block text-accent underline"
          href={`${GITHUB_URL}/tree/main/notebooks`}
        >
          {t("guide.notebooks")}
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
