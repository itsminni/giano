import { Footer } from "./Footer";
import { Guide } from "./Guide";
import { Home } from "./Home";
import { Impact } from "./Impact";
import { Story } from "./Story";
import { useI18n } from "./i18n";

const sections = ["home", "impatto", "storia", "guida"] as const;

export function ReadableSite() {
  const { t } = useI18n();
  return (
    <div className="min-h-screen">
      <a href="#main-content" className="sr-only focus:not-sr-only">
        {t("site.skip")}
      </a>
      <header className="mx-auto max-w-6xl px-3 pt-5 sm:px-6">
        <nav className="glass-strong flex flex-wrap items-center gap-3 rounded-3xl p-4 sm:px-6">
          <a href="#home" className="mr-auto font-display text-lg font-bold">
            Giano
          </a>
          {sections.map((section) => (
            <a
              key={section}
              href={`#${section}`}
              className="rounded-full px-3 py-2 text-sm text-accent hover:underline"
            >
              {t(`nav.${section}`)}
            </a>
          ))}
        </nav>
      </header>
      <main
        id="main-content"
        className="mx-auto max-w-6xl space-y-20 px-3 py-10 sm:px-6"
      >
        <section id="home" className="scroll-mt-6">
          <Home />
        </section>
        <section id="impatto" className="scroll-mt-6">
          <Impact />
        </section>
        <section id="storia" className="scroll-mt-6">
          <Story />
        </section>
        <section id="guida" className="scroll-mt-6">
          <Guide />
        </section>
      </main>
      <Footer />
    </div>
  );
}
