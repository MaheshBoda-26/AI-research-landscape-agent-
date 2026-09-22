"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { StageTimeline, initialStageState, reduceStage, type StageState } from "@/components/StageTimeline";
import { streamLandscape } from "@/lib/api";
import type { StageName, StreamError } from "@/lib/types";

export type RunState = {
  running: boolean;
  stages: Record<StageName, StageState>;
  error: StreamError | null;
  /** Survives a retryable failure so the retry button can reuse the topic. */
  topic: string;
};

const EMPTY: RunState = {
  running: false,
  stages: initialStageState(),
  error: null,
  topic: "",
};

export function RunView({
  topic,
  active,
  onFinished,
}: {
  topic: string;
  active: boolean;
  onFinished: (landscapeId: number) => void;
}) {
  const [state, setState] = useState<RunState>(EMPTY);
  const abortRef = useRef<AbortController | null>(null);
  const startedRef = useRef<string | null>(null);

  const start = useCallback(
    (value: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({
        running: true,
        stages: initialStageState(),
        error: null,
        topic: value,
      });

      streamLandscape(
        value,
        {
          onStage: (event) =>
            setState((previous) => ({
              ...previous,
              stages: reduceStage(previous.stages, event),
            })),
          onError: (error) =>
            setState((previous) => ({ ...previous, running: false, error })),
          onDone: (done) => {
            setState((previous) => ({ ...previous, running: false }));
            onFinished(done.landscape_id);
          },
        },
        controller.signal,
      ).catch((error: unknown) => {
        // An abort is a navigation, not a failure worth showing.
        if (error instanceof DOMException && error.name === "AbortError") return;
        setState((previous) => ({
          ...previous,
          running: false,
          error: {
            run_id: "",
            landscape_id: null,
            stage: "retrieval",
            message: error instanceof Error ? error.message : String(error),
            retryable: true,
          },
        }));
      });
    },
    [onFinished],
  );

  useEffect(() => {
    if (!active) return;
    if (startedRef.current === topic) return;
    startedRef.current = topic;
    start(topic);
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
      // Allow the same topic to be re-run deliberately later.
      startedRef.current = null;
    };
  }, [active, topic, start]);

  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-10">
      <div className="mb-5">
        <p className="text-xs uppercase tracking-wider text-[var(--color-muted)]">
          Building landscape
        </p>
        <h1 className="mt-1 text-xl font-semibold">{state.topic}</h1>
      </div>

      <StageTimeline stages={state.stages} />

      {state.error && (
        <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/5 p-4">
          <p className="text-sm font-medium text-red-300">
            {state.error.retryable ? "This looks temporary" : "The run failed"}
          </p>
          <p className="mt-1 text-xs text-[var(--color-muted)]">
            {state.error.message}
          </p>
          {state.error.retryable && (
            <button
              type="button"
              onClick={() => start(state.topic)}
              className="mt-3 rounded-md bg-[var(--color-accent)]/15 px-3 py-1.5 text-xs font-medium text-[var(--color-accent)] hover:bg-[var(--color-accent)]/25"
            >
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}
