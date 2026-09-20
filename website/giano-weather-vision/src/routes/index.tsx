import { createFileRoute, stripSearchParams } from "@tanstack/react-router";
import { Nav, TABS, type TabId } from "@/components/giano/Nav";
import { Home } from "@/components/giano/Home";
import { MapSection } from "@/components/giano/MapSection";
import { Impact } from "@/components/giano/Impact";
import { Story } from "@/components/giano/Story";
import { Demo } from "@/components/giano/Demo";
import { Guide } from "@/components/giano/Guide";
import { Footer } from "@/components/giano/Footer";
import { LanguageProvider, useI18n } from "@/components/giano/i18n";

const title = "Giano — Ricostruzione dei dati meteo del Trentino";
const description =
  "Giano ricostruisce osservazioni meteorologiche mancanti. Esplora le serie di 171 stazioni Meteotrentino, gli esempi di ricostruzione e la guida all'uso.";

export const Route = createFileRoute("/")({
  validateSearch: (search: Record<string, unknown>) => ({
    tab: TABS.some((item) => item.id === search["tab"])
      ? (search["tab"] as TabId)
      : ("home" as TabId),
    station: typeof search["station"] === "string" ? search["station"] : "",
    variable: typeof search["variable"] === "string" ? search["variable"] : "",
    mode:
      search["mode"] === "history"
        ? ("history" as const)
        : ("example" as const),
  }),
  search: {
    middlewares: [
      stripSearchParams({
        tab: "home",
        station: "",
        variable: "",
        mode: "example",
      }),
    ],
  },
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <LanguageProvider>
      <Site />
    </LanguageProvider>
  );
}

function Site() {
  const { tab, station, variable, mode } = Route.useSearch();
  const navigate = Route.useNavigate();
  const { t } = useI18n();

  const go = (t: TabId) => {
    void navigate({
      search: { tab: t, station: "", variable: "", mode: "example" },
    });
    if (typeof window !== "undefined")
      window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-xl focus:bg-background focus:p-4"
      >
        {t("site.skip")}
      </a>
      <Nav active={tab} onChange={go} />
      <main
        id="main-content"
        className="mx-auto max-w-6xl px-3 pb-10 pt-28 sm:px-6 sm:pt-36"
      >
        <div
          key={tab}
          className="animate-in fade-in slide-in-from-bottom-2 duration-300"
        >
          {tab === "home" && <Home onNavigate={go} />}
          {tab === "mappa" && (
            <MapSection
              stationId={station}
              variableKey={variable}
              mode={mode}
              onSelection={(nextStation, nextVariable, nextMode) => {
                void navigate({
                  search: (previous) => ({
                    ...previous,
                    station: nextStation,
                    variable: nextVariable,
                    mode: nextMode,
                  }),
                  resetScroll: false,
                });
              }}
            />
          )}
          {tab === "impatto" && <Impact onNavigate={go} />}
          {tab === "storia" && <Story />}
          {tab === "demo" && <Demo />}
          {tab === "guida" && <Guide />}
        </div>
      </main>
      <Footer />
    </div>
  );
}
