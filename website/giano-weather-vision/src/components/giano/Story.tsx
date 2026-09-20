import { BookOpen } from "lucide-react";
import { team, webvalleyTutor } from "./data";
import { useI18n } from "./i18n";

function initials(name: string) {
  return name
    .split(" ")
    .map((p) => p[0])
    .slice(0, 2)
    .join("");
}

export function Story() {
  const { t } = useI18n();

  return (
    <div className="space-y-14">
      <div className="max-w-2xl">
        <h2 className="font-display text-4xl font-bold sm:text-5xl">
          {t("story.title")}
        </h2>
        <p className="mt-3 text-muted-foreground">{t("story.intro")}</p>
      </div>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="glass rounded-[2rem] p-8">
          <BookOpen className="h-6 w-6 text-accent" />
          <h3 className="mt-3 font-display text-2xl font-bold">
            {t("story.genesis")}
          </h3>
          <p className="mt-3 text-muted-foreground">{t("story.g1")}</p>
          <p className="mt-3 text-muted-foreground">{t("story.g2")}</p>
          <div className="mt-6 space-y-2 text-sm">
            <p className="font-medium">{t("story.sources")}</p>
            <a
              className="block text-accent underline"
              href="https://webvalley.fbk.eu/challenge/"
            >
              {t("story.sourceSchool")}
            </a>
            <a
              className="block text-accent underline"
              href="https://magazine.fbk.eu/en/news/challenges-that-make-you-grow/"
            >
              {t("story.sourceMagazine")}
            </a>
          </div>
        </div>

        <div className="glass overflow-hidden rounded-[2rem] p-3">
          <div className="aspect-video w-full overflow-hidden rounded-[1.5rem]">
            <iframe
              src="https://www.youtube.com/embed/A7KIoiKSjnU"
              title={t("story.videoTitle")}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="h-full w-full"
            />
          </div>
          <p className="px-4 py-3 text-sm text-muted-foreground">
            {t("story.videoCaption")}
          </p>
        </div>
      </section>

      <section className="glass rounded-[2rem] p-6 sm:p-8">
        <h3 className="font-display text-2xl font-bold">
          {t("story.prototypeTitle")}
        </h3>
        <p className="mt-4 max-w-4xl leading-relaxed text-muted-foreground">
          {t("story.prototype1")}
        </p>
        <p className="mt-4 max-w-4xl leading-relaxed text-muted-foreground">
          {t("story.prototype2")}
        </p>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="glass rounded-[2rem] p-6 sm:p-8">
          <h3 className="font-display text-2xl font-bold">
            {t("story.evolutionTitle")}
          </h3>
          <p className="mt-4 leading-relaxed text-muted-foreground">
            {t("story.evolution1")}
          </p>
          <p className="mt-4 leading-relaxed text-muted-foreground">
            {t("story.evolution2")}
          </p>
        </div>
        <div className="glass rounded-[2rem] p-6 sm:p-8">
          <h3 className="font-display text-2xl font-bold">
            {t("story.releaseTitle")}
          </h3>
          <p className="mt-4 leading-relaxed text-muted-foreground">
            {t("story.release1")}
          </p>
          <p className="mt-4 leading-relaxed text-muted-foreground">
            {t("story.release2")}
          </p>
        </div>
      </section>

      <section>
        <h3 className="font-display text-2xl font-bold">{t("story.team")}</h3>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[...team, webvalleyTutor].map((name) => (
            <article key={name} className="glass rounded-3xl p-6">
              <span
                aria-hidden="true"
                className="grid h-14 w-14 place-items-center rounded-2xl bg-primary/20 font-display text-lg font-bold text-primary"
              >
                {initials(name)}
              </span>
              <h4 className="mt-4 font-display text-lg font-semibold">
                {name}
              </h4>
              <p className="mt-3 text-xs text-muted-foreground">
                {t(name === webvalleyTutor ? "story.tutor" : "story.member")}
              </p>
            </article>
          ))}
        </div>
        <p className="mt-4 text-sm text-muted-foreground">
          {t("story.tutorNote")}{" "}
          <a
            className="text-accent underline"
            href="https://webvalley.fbk.eu/team/"
            target="_blank"
            rel="noreferrer"
          >
            {t("story.tutorList")}
          </a>
        </p>
      </section>

      <section className="border-t border-foreground/15 pt-8">
        <h3 className="font-display text-2xl font-bold">
          {t("story.thanksTitle")}
        </h3>
        <p className="mt-4 max-w-3xl leading-relaxed text-muted-foreground">
          {t("story.thanksData")}
        </p>
        <p className="mt-3 max-w-3xl leading-relaxed text-muted-foreground">
          {t("story.thanksSchool")}
        </p>
        <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-sm text-accent underline">
          <a href="https://www.meteotrentino.it/">Meteotrentino</a>
          <a href="https://www.fbk.eu/">Fondazione Bruno Kessler</a>
          <a href="https://webvalley.fbk.eu/">WebValley</a>
        </div>
      </section>
    </div>
  );
}
