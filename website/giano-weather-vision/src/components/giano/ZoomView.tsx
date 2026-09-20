import { useId, useRef, useState, type ReactNode } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";
import { useI18n } from "./i18n";

const LEVELS = [1, 1.5, 2, 3, 4];

export function ZoomView({
  children,
  label,
  map = false,
  length = 0,
}: {
  children: ReactNode | ((start: number, end: number) => ReactNode);
  label: string;
  map?: boolean;
  length?: number;
}) {
  const { t } = useI18n();
  const [level, setLevel] = useState(0);
  const [position, setPosition] = useState(50);
  const viewport = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
    moved: boolean;
  } | null>(null);
  const id = useId();
  const zoom = LEVELS[level]!;
  const series = typeof children === "function";
  const visible = Math.max(2, Math.ceil(length / zoom));
  const start = Math.round((Math.max(0, length - visible) * position) / 100);

  function change(next: number) {
    if (series) {
      setLevel(next);
      if (!next) setPosition(50);
      return;
    }
    const element = viewport.current;
    if (!element) return;
    const ratio = LEVELS[next]! / zoom;
    const left =
      (element.scrollLeft + element.clientWidth / 2) * ratio -
      element.clientWidth / 2;
    const top = map
      ? (element.scrollTop + element.clientHeight / 2) * ratio -
        element.clientHeight / 2
      : 0;
    setLevel(next);
    requestAnimationFrame(() =>
      element.scrollTo({ left: Math.max(0, left), top: Math.max(0, top) }),
    );
  }

  return (
    <div className="min-w-0 space-y-2" role="group" aria-label={label}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p id={`${id}-hint`} className="text-xs text-muted-foreground">
          {t(level ? (series ? "zoom.rangeHint" : "zoom.pan") : "zoom.hint")}
        </p>
        <div className="glass inline-flex shrink-0 items-center rounded-full p-1">
          <button
            type="button"
            aria-label={t("zoom.out")}
            aria-controls={id}
            disabled={level === 0}
            onClick={() => change(level - 1)}
            className="flex h-11 w-11 items-center justify-center rounded-full hover:bg-foreground/10 disabled:opacity-30"
          >
            <Minus aria-hidden="true" size={18} />
          </button>
          <output
            className="w-12 text-center text-xs tabular-nums"
            aria-live="polite"
          >
            {zoom * 100}%
          </output>
          <button
            type="button"
            aria-label={t("zoom.in")}
            aria-controls={id}
            disabled={level === LEVELS.length - 1 || (series && visible <= 2)}
            onClick={() => change(level + 1)}
            className="flex h-11 w-11 items-center justify-center rounded-full hover:bg-foreground/10 disabled:opacity-30"
          >
            <Plus aria-hidden="true" size={18} />
          </button>
          <button
            type="button"
            aria-label={t("zoom.reset")}
            aria-controls={id}
            disabled={level === 0}
            onClick={() => change(0)}
            className="flex h-11 w-11 items-center justify-center rounded-full text-accent hover:bg-foreground/10 disabled:opacity-30"
          >
            <RotateCcw aria-hidden="true" size={17} />
          </button>
        </div>
      </div>
      {series && level > 0 && (
        <label className="flex items-center gap-3 text-xs text-muted-foreground">
          {t("zoom.range")}
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={position}
            onChange={(event) => setPosition(Number(event.target.value))}
            aria-label={t("zoom.range")}
            className="h-11 min-w-0 flex-1 accent-[var(--accent)]"
          />
        </label>
      )}
      <div
        ref={viewport}
        id={id}
        tabIndex={0}
        role="region"
        aria-label={label}
        aria-describedby={`${id}-hint`}
        className={`overflow-auto rounded-2xl ${map ? "h-[300px] sm:h-[420px]" : ""} ${level && !series ? "cursor-grab active:cursor-grabbing" : ""}`}
        onPointerDown={(event) => {
          if (
            series ||
            event.pointerType !== "mouse" ||
            event.button !== 0 ||
            !level
          )
            return;
          drag.current = {
            x: event.clientX,
            y: event.clientY,
            left: event.currentTarget.scrollLeft,
            top: event.currentTarget.scrollTop,
            moved: false,
          };
        }}
        onPointerMove={(event) => {
          const start = drag.current;
          if (!start || event.buttons !== 1) return;
          const dx = event.clientX - start.x;
          const dy = event.clientY - start.y;
          if (Math.abs(dx) + Math.abs(dy) < 5) return;
          start.moved = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.scrollTo({
            left: start.left - dx,
            top: map ? start.top - dy : 0,
          });
          event.preventDefault();
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onClickCapture={(event) => {
          if (drag.current?.moved) {
            event.preventDefault();
            event.stopPropagation();
          }
          drag.current = null;
        }}
      >
        <div
          style={{
            width: `${series ? 100 : zoom * 100}%`,
            ...(map ? { height: `${zoom * 100}%` } : {}),
          }}
        >
          {typeof children === "function"
            ? children(start, start + visible)
            : children}
        </div>
      </div>
    </div>
  );
}
