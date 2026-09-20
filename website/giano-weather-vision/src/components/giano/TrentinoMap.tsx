import { stations, type Station } from "./data";
import { useI18n } from "./i18n";
import { projectToMap, TRENTINO_BOUNDARY_PATH } from "./trentinoBoundary";
import { ZoomView } from "./ZoomView";

export function TrentinoMap({
  selected,
  onSelect,
  compact = false,
  list = stations,
}: {
  selected?: Station | null;
  onSelect?: (s: Station) => void;
  compact?: boolean;
  list?: Station[];
}) {
  const { t } = useI18n();

  return (
    <div className="min-w-0 space-y-3">
      <ZoomView label={t("map.ariaBoundary")} map>
        <div className="glass relative h-full w-full overflow-hidden rounded-3xl">
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="absolute inset-0 h-full w-full"
            aria-label={t("map.ariaBoundary")}
          >
            <defs>
              <linearGradient id="trentino-terrain" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="var(--cyan)" stopOpacity="0.25" />
                <stop
                  offset="100%"
                  stopColor="var(--deep)"
                  stopOpacity="0.55"
                />
              </linearGradient>
              <clipPath id="trentino-clip">
                <path d={TRENTINO_BOUNDARY_PATH} />
              </clipPath>
            </defs>
            <path
              d={TRENTINO_BOUNDARY_PATH}
              fill="url(#trentino-terrain)"
              stroke="var(--cyan)"
              strokeOpacity="0.78"
              strokeWidth="0.45"
              vectorEffect="non-scaling-stroke"
            />
            <g clipPath="url(#trentino-clip)" opacity="0.42">
              {[18, 29, 40, 51, 62, 73, 84].map((y, index) => (
                <path
                  key={y}
                  d={`M2 ${y} Q 22 ${y - 8 + (index % 2) * 3}, 43 ${y - 1} T 72 ${y + 2} T 98 ${y - 4}`}
                  fill="none"
                  stroke="var(--aqua)"
                  strokeOpacity="0.34"
                  strokeWidth="0.32"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              <path
                d="M17 69 C 29 62, 34 52, 43 44 S 57 31, 68 24"
                fill="none"
                stroke="var(--primary)"
                strokeOpacity="0.34"
                strokeWidth="0.42"
                strokeDasharray="1.5 1.2"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          </svg>

          {!compact && (
            <div className="pointer-events-none absolute inset-0 text-[9px] text-muted-foreground/70">
              <span className="absolute left-[40%] top-[58%]">Trento</span>
              <span className="absolute left-[36%] top-[76%]">Rovereto</span>
              <span className="absolute left-[64%] top-[38%]">Valsugana</span>
              <span className="absolute left-[23%] top-[29%]">Val di Sole</span>
              <span className="absolute left-[68%] top-[17%]">
                Val di Fassa
              </span>
            </div>
          )}

          {list.map((s) => {
            const isSel = selected?.id === s.id;
            const size = compact ? 5 : 7;
            const position = projectToMap(s.lon, s.lat);
            return (
              <button
                key={s.id}
                onClick={() => onSelect?.(s)}
                disabled={!onSelect}
                style={{
                  left: `${position.x}%`,
                  top: `${position.y}%`,
                  width: 24,
                  height: 24,
                }}
                className="group absolute flex -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full"
                aria-label={s.name}
                aria-pressed={onSelect ? isSel : undefined}
                title={s.name}
              >
                <span
                  style={{ width: size, height: size }}
                  className={`block rounded-full transition-all ${
                    isSel
                      ? "scale-[2] bg-accent ring-2 ring-accent/50"
                      : s.hasHistory
                        ? "bg-accent/70 group-hover:scale-150"
                        : "bg-foreground/25 group-hover:scale-150"
                  } ${s.hasValidation ? "ring-1 ring-foreground ring-offset-2 ring-offset-background" : ""}`}
                />
                {!compact && (
                  <span className="pointer-events-none absolute left-1/2 top-full z-10 mt-1 -translate-x-1/2 whitespace-nowrap rounded-full bg-popover px-2 py-0.5 text-[10px] text-popover-foreground opacity-0 transition-opacity group-hover:opacity-100">
                    {s.name}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </ZoomView>
      <div
        className="relative flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-border pt-3 text-xs text-muted-foreground"
        aria-label={t("map.legendTitle")}
      >
        <span>{t("map.stationsCount", { n: list.length })}</span>
        <span className="inline-flex items-center gap-1">
          <i className="h-2 w-2 rounded-full bg-accent/70" />{" "}
          {t("map.legend.giano")}
        </span>
        <span className="inline-flex items-center gap-1">
          <i className="h-2 w-2 rounded-full ring-1 ring-foreground" />{" "}
          {t("map.legend.local")}
        </span>
        <span className="inline-flex items-center gap-1">
          <i className="h-2 w-2 rounded-full bg-foreground/25" />{" "}
          {t("map.legend.catalog")}
        </span>
        <span className="w-full">{t("map.legend.boundary")}</span>
      </div>
    </div>
  );
}
