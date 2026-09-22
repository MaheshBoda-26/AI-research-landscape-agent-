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
      <span className="grid h-5 w-5 place-items-center rounded-full bg-emerald-500/15 text-[11px] text-emerald-400">
        ✓
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="stage-running grid h-5 w-5 place-items-center rounded-full bg-[var(--color-accent)]/20 text-[10px] text-[var(--color-accent)]">
        ●
      </span>
    );
  }
  if (status === "error") {
    return (
      <span className="grid h-5 w-5 place-items-center rounded-full bg-red-500/15 text-[11px] text-red-400">
        !
      </span>
    );
  }
  return (
    <span className="grid h-5 w-5 place-items-center rounded-full border border-[var(--color-edge)] text-[10px] text-[var(--color-muted)]">
      ·
    </span>
  );
}

export function StageTimeline({
  stages,
}: {
  stages: Record<StageName, StageState>;
}) {
  return (
    <ol className="divide-y divide-[var(--color-edge)] overflow-hidden rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)]">
      {STAGE_ORDER.map((stage, index) => {
        const state = stages[stage];
        const showBar = state.status === "running" && state.total > 0;
        const pct = showBar
          ? Math.max(4, Math.round((state.current / state.total) * 100))
          : 0;

        return (
          <li key={stage} className="px-4 py-3">
            <div className="flex items-center gap-3">
              <StatusDot status={state.status} />
              <span className="w-5 text-[11px] tabular-nums text-[var(--color-muted)]">
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
              </span>
              {state.total > 0 && (
                <span className="ml-auto text-[11px] tabular-nums text-[var(--color-muted)]">
                  {state.current}/{state.total}
                </span>
              )}
            </div>
            {state.message && (
              <p className="mt-1 pl-[52px] text-xs text-[var(--color-muted)]">
                {state.message}
              </p>
            )}
            {showBar && (
              <div className="mt-2 ml-[52px] h-1 overflow-hidden rounded-full bg-[var(--color-panel-2)]">
                <div
                  className="h-full rounded-full bg-[var(--color-accent)] transition-[width] duration-300"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
