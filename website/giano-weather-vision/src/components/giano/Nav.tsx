import { Github, Menu } from "lucide-react";
import { useState } from "react";
import { GITHUB_URL } from "./data";
import { useI18n, type Lang } from "./i18n";

export const TABS = [
  { id: "home" },
  { id: "mappa" },
  { id: "impatto" },
  { id: "storia" },
  { id: "demo" },
  { id: "guida" },
] as const;

export type TabId = (typeof TABS)[number]["id"];

function LangSwitch({ className = "" }: { className?: string }) {
  const { lang, setLang, t } = useI18n();
  return (
    <div
      className={`glass inline-flex items-center rounded-full p-0.5 ${className}`}
      role="group"
      aria-label={t("nav.lang")}
    >
      {(["it", "en"] as Lang[]).map((l) => (
        <button
          key={l}
          onClick={() => setLang(l)}
          aria-pressed={lang === l}
          className={`rounded-full px-3 py-1.5 text-xs font-semibold uppercase transition-colors ${
            lang === l
              ? "bg-primary/25 text-primary ring-1 ring-primary/40"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {l}
        </button>
      ))}
    </div>
  );
}

export function Nav({
  active,
  onChange,
}: {
  active: TabId;
  onChange: (t: TabId) => void;
}) {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();

  return (
    <header className="fixed inset-x-0 top-0 z-50 px-3 pt-3 sm:px-6 sm:pt-5">
      <nav className="glass-strong mx-auto flex max-w-6xl items-center gap-3 rounded-3xl px-4 py-3 sm:px-6">
        <button
          onClick={() => onChange("home")}
          className="flex min-w-0 items-center gap-2 text-left"
        >
          <img
            src={`${import.meta.env.BASE_URL ?? "/"}logo.png`}
            alt="Logo Giano"
            className="h-9 w-9 shrink-0 object-contain"
          />
          <span className="truncate font-display text-lg font-bold tracking-tight">
            Giano
          </span>
        </button>

        <div className="ml-auto hidden items-center gap-1 lg:flex">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              aria-current={active === tab.id ? "page" : undefined}
              className={`rounded-full px-4 py-2 text-sm font-medium transition-all ${
                active === tab.id
                  ? "bg-primary/20 text-primary ring-1 ring-primary/40"
                  : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
              }`}
            >
              {t(`nav.${tab.id}`)}
            </button>
          ))}
          <LangSwitch className="ml-2" />
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="ml-2 inline-flex items-center gap-2 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground transition-transform hover:scale-[1.03]"
          >
            <Github className="h-4 w-4" /> GitHub
          </a>
        </div>

        <div className="ml-auto flex items-center gap-2 lg:hidden">
          <LangSwitch />
          <button
            className="rounded-full p-2 text-foreground"
            onClick={() => setOpen((v) => !v)}
            aria-label={t("nav.menu")}
            aria-expanded={open}
            aria-controls="mobile-navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
        </div>
      </nav>

      {open && (
        <div
          id="mobile-navigation"
          className="glass-strong mx-auto mt-2 grid max-w-6xl gap-1 rounded-3xl p-3 lg:hidden"
        >
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                onChange(tab.id);
                setOpen(false);
              }}
              className={`rounded-2xl px-4 py-2 text-left text-sm ${
                active === tab.id
                  ? "bg-primary/20 text-primary"
                  : "text-muted-foreground"
              }`}
            >
              {t(`nav.${tab.id}`)}
            </button>
          ))}
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="mt-1 inline-flex items-center gap-2 rounded-2xl bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground"
          >
            <Github className="h-4 w-4" /> GitHub
          </a>
        </div>
      )}
    </header>
  );
}
