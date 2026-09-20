import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import it from "../../content/it";
import en from "../../content/en";

export type Lang = "it" | "en";
const dict: Record<Lang, Record<string, string>> = { it, en };
function browserLanguage(): Lang {
  if (typeof navigator === "undefined") return "it";
  const languages = navigator.languages?.length
    ? navigator.languages
    : [navigator.language];
  const preferredLanguage = languages[0] ?? "";
  return preferredLanguage.toLowerCase().startsWith("it") ? "it" : "en";
}

function initialLanguage(): Lang {
  try {
    const stored = localStorage.getItem("giano-lang");
    if (stored === "it" || stored === "en") return stored;
  } catch {
    /* Storage unavailable. */
  }
  return browserLanguage();
}

type Ctx = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
  locale: string;
};
const LanguageContext = createContext<Ctx | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, updateLang] = useState<Lang>("it");
  useEffect(() => {
    updateLang(initialLanguage());
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);
  const value = useMemo<Ctx>(
    () => ({
      lang,
      locale: dict[lang]["locale"] ?? "it-IT",
      setLang: (next) => {
        updateLang(next);
        try {
          localStorage.setItem("giano-lang", next);
        } catch {
          /* Storage unavailable. */
        }
      },
      t: (key, vars) => {
        let out = dict[lang][key] ?? dict.it[key] ?? key;
        for (const [k, v] of Object.entries(vars ?? {}))
          out = out.split(`{${k}}`).join(String(v));
        return out;
      },
    }),
    [lang],
  );
  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}
export function useI18n() {
  const context = useContext(LanguageContext);
  if (!context) throw new Error("useI18n must be used inside LanguageProvider");
  return context;
}
