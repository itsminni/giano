import { Github, Mail } from "lucide-react";
import { GITHUB_URL } from "./data";
import { useI18n } from "./i18n";

export function Footer() {
  const { t } = useI18n();
  return (
    <footer className="mx-auto mt-12 w-full max-w-6xl px-3 pb-6 sm:px-6">
      <div className="glass rounded-[2rem] p-6 sm:p-8">
        <div className="grid gap-8 sm:grid-cols-[1fr_auto]">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <img
                src={`${import.meta.env.BASE_URL ?? "/"}logo.png`}
                alt="Logo Giano"
                className="h-5 w-5 object-contain"
              />
              <span className="font-display text-lg font-bold">Giano</span>
            </div>
            <p className="mt-2 max-w-md text-sm text-muted-foreground">
              {t("footer.text")}
            </p>
            <a
              href="mailto:team.giano2025@gmail.com"
              className="mt-3 inline-flex items-center gap-2 text-sm text-accent hover:underline"
            >
              <Mail className="h-4 w-4" /> team.giano2025@gmail.com
            </a>
          </div>
          <div className="flex flex-col items-start gap-3 sm:items-end">
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className="glass inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm"
            >
              <Github className="h-4 w-4" /> GitHub
            </a>
            <span className="text-sm text-muted-foreground">
              {t("footer.badge")}
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
