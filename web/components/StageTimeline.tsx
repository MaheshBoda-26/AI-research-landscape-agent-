"use client";

import { STAGE_LABELS, STAGE_ORDER, type StageEvent, type StageName } from "@/lib/types";

export type StageState = {
  status: "pending" | "running" | "done" | "error";
  message: string;
  current: number;
  total: number;
};

export const initialStageState = (): Record<StageName, StageState> =>
  Object.fromEntries(
    STAGE_ORDER.map((stage) => [
      stage,
      { status: "pending", message: "", current: 0, total: 0 },
    ]),
  ) as Record<StageName, StageState>;

export function reduceStage(
  state: Record<StageName, StageState>,
  event: StageEvent,
): Record<StageName, StageState> {
  const previous = state[event.stage];
  // A later "running" tick must not overwrite a "done", and nothing may
  // overwrite an "error": events arrive in order, but progress ticks repeat.
  if (previous.status === "done" && event.status === "running") return state;
  if (previous.status === "error") return state;

  return {
    ...state,
    [event.stage]: {
      status: event.status,
      message: event.message || previous.message,
      current: event.progress?.current ?? 0,
      total: event.progress?.total ?? previous.total,
    },
  };
}

function StatusDot({ status }: { status: StageState["status"] }) {
  if (status === "done") {
    return (
      <span
        aria-hidden="true"
        className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--color-accent)]/15 text-[11px] text-[var(--color-accent)]"
      >
        ✓
      </span>
    );
  }
  if (status === "running") {
    return (
      <span
        aria-hidden="true"
        className="relative grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--color-accent)]/20 text-[9px] text-[var(--color-accent)]"
      >
        ●
        <span className="absolute inset-0 animate-ping rounded-full bg-[var(--color-accent)]/20 motion-reduce:animate-none" />
      </span>
    );
  }
  if (status === "error") {
    return (
      <span
        aria-hidden="true"
        className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--color-danger)]/15 text-[11px] text-[var(--color-danger)]"
      >
        !
      </span>
    );
  }
  return (
    <span
      aria-hidden="true"
      className="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-[var(--color-edge)] text-[10px] text-[var(--color-muted)]"
    >
      ·
    </span>
  );
}

/**
 * The five pipeline stages. `borderless` embeds the timeline inside an
 * already-carded container (the expand strip) without nesting cards.
 */
export function StageTimeline({
  stages,
  borderless = false,
}: {
  stages: Record<StageName, StageState>;
  borderless?: boolean;
}) {
  return (
    <ol
      aria-live="polite"
      className={
        borderless
          ? "divide-y divide-[var(--color-edge)]"
          : "divide-y divide-[var(--color-edge)] overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-edge)] bg-[var(--color-panel)]"
      }
    >
      {STAGE_ORDER.map((stage, index) => {
        const state = stages[stage];
        const showBar = state.status === "running" && state.total > 0;
        const pct = showBar
          ? Math.max(4, Math.round((state.current / state.total) * 100))
          : 0;
        const statusText =
          state.status === "done"
            ? "completed"
            : state.status === "running"
              ? "in progress"
              : state.status === "error"
                ? "failed"
                : "pending";

        return (
          <li key={stage} className="px-4 py-3">
            <div className="flex items-center gap-3">
              <StatusDot status={state.status} />
              <span
                aria-hidden="true"
                className="w-4 text-xs tabular-nums text-[var(--color-muted)]"
              >
                {index + 1}
              </span>
              <span
                className={`text-sm font-medium ${
                  state.status === "pending"
                    ? "text-[var(--color-muted)]"
                    : "text-[var(--color-ink)]"
                }`}
              >
                {STAGE_LABELS[stage]}
                <span className="visually-hidden"> — {statusText}</span>
              </span>
              {state.total > 0 && (
                <span className="ml-auto text-xs tabular-nums text-[var(--color-muted)]">
                  {state.current}/{state.total}
                </span>
              )}
            </div>
            {state.message && (
              <p className="mt-1 pl-[48px] text-xs leading-relaxed text-[var(--color-muted)]">
                {state.message}
              </p>
            )}
            {showBar && (
              <div
                className="ml-[48px] mt-2 h-1 overflow-hidden rounded-full bg-[var(--color-panel-2)]"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={state.total}
                aria-valuenow={state.current}
                aria-label={`${STAGE_LABELS[stage]} progress`}
              >
                {/* scaleX, not width: compositing only, never layout. */}
                <div
                  className="h-full origin-left rounded-full bg-[var(--color-accent)] transition-transform duration-300 ease-out"
                  style={{ transform: `scaleX(${pct / 100})` }}
                />
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
