"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";

import { LandscapePanels } from "@/components/LandscapePanels";
import { MapCanvas } from "@/components/MapCanvas";
import { PaperPanel } from "@/components/PaperPanel";
import { StageTimeline, initialStageState, reduceStage, type StageState } from "@/components/StageTimeline";
import { expandLandscape, getLandscape } from "@/lib/api";
import type { LandscapeDetail, StageName } from "@/lib/types";

export default function LandscapePage() {
  const params = useParams<{ id: string }>();
  const landscapeId = Number(params.id);

  const [landscape, setLandscape] = useState<LandscapeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  const [expanding, setExpanding] = useState(false);
  const [stages, setStages] = useState<Record<StageName, StageState>>(initialStageState);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      try {
        setLandscape(await getLandscape(landscapeId, signal));
        setError(null);
      } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(cause instanceof Error ? cause.message : "Could not load.");
      }
    },
    [landscapeId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const onExpand = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setExpanding(true);
    setStages(initialStageState());

    expandLandscape(
      landscapeId,
      {
        onStage: (event) =>
          setStages((previous) => reduceStage(previous, event)),
        onError: (streamError) => {
          setExpanding(false);
          setError(streamError.message);
        },
        onDone: () => {
          setExpanding(false);
          void load();
        },
      },
      controller.signal,
    ).catch((cause: unknown) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setExpanding(false);
      setError(cause instanceof Error ? cause.message : "Expansion failed.");
    });
  }, [landscapeId, load]);

  const selectedPaper = useMemo(
    () =>
      landscape?.papers.find((paper) => paper.paper_id === selectedPaperId) ?? null,
    [landscape, selectedPaperId],
  );

  if (error && !landscape) {
    return (
      <main className="mx-auto max-w-xl px-6 py-20 text-center">
        <p className="text-sm text-red-300">{error}</p>
        <a href="/" className="mt-4 inline-block text-xs text-[var(--color-accent)]">
          ← Back to topics
        </a>
      </main>
    );
  }

  if (!landscape) {
    return (
      <main className="mx-auto max-w-xl px-6 py-20 text-center text-sm text-[var(--color-muted)]">
        Loading landscape…
      </main>
    );
  }

  return (
    <main className="flex h-[calc(100vh-53px)] flex-col">
      <div className="flex items-start gap-4 border-b border-[var(--color-edge)] bg-[var(--color-panel)]/50 px-5 py-3">
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-base font-semibold">{landscape.title}</h1>
          <p className="mt-0.5 text-[11px] text-[var(--color-muted)]">
            {landscape.papers.length} papers ·{" "}
            {landscape.clusters.filter((cluster) => !cluster.is_unclustered).length}{" "}
            clusters · generation {landscape.generation} · {landscape.topic}
          </p>
        </div>
        <button
          type="button"
          onClick={onExpand}
          disabled={expanding}
          className="shrink-0 rounded-md border border-[var(--color-edge)] px-3 py-1.5 text-xs text-[var(--color-muted)] transition-colors hover:border-[var(--color-accent)]/50 hover:text-[var(--color-ink)] disabled:opacity-40"
        >
          {expanding ? "Growing…" : "Grow this map"}
        </button>
      </div>

      {expanding && (
        <div className="border-b border-[var(--color-edge)] px-5 py-3">
          <p className="mb-2 text-[11px] text-[var(--color-muted)]">
            Pulling in more papers. Existing papers stay put — the layout is only
            recomputed once, at the end.
          </p>
          <StageTimeline stages={stages} />
        </div>
      )}

      {landscape.summary && (
        <p className="border-b border-[var(--color-edge)] px-5 py-2.5 text-xs leading-relaxed text-[var(--color-muted)]">
          {landscape.summary}
        </p>
      )}

      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1">
          <MapCanvas
            landscape={landscape}
            selectedPaperId={selectedPaperId}
            onSelectPaper={setSelectedPaperId}
          />
        </div>
        <div className="w-[350px] shrink-0 border-l border-[var(--color-edge)]">
          {selectedPaper ? (
            <PaperPanel
              paper={selectedPaper}
              onClose={() => setSelectedPaperId(null)}
            />
          ) : (
            <LandscapePanels
              landscape={landscape}
              onSelectPaper={setSelectedPaperId}
            />
          )}
        </div>
      </div>
    </main>
  );
}
