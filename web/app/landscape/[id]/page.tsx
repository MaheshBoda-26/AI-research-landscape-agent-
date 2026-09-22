"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

import { LandscapePanels } from "@/components/LandscapePanels";
import { MapCanvas } from "@/components/MapCanvas";
import { PaperPanel } from "@/components/PaperPanel";
import {
  StageTimeline,
  initialStageState,
  reduceStage,
  type StageState,
} from "@/components/StageTimeline";
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

  useEffect(() => {
    // Cancel any in-flight expansion when the page unmounts.
    return () => abortRef.current?.abort();
  }, []);

  const onExpand = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setExpanding(true);
    setStages(initialStageState());

    expandLandscape(
      landscapeId,
      {
        onStage: (event) => setStages((previous) => reduceStage(previous, event)),
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

  if (Number.isNaN(landscapeId)) {
    return (
      <main className="app-main mx-auto max-w-xl px-6 py-20 text-center">
        <h1 className="text-xl font-semibold">Invalid landscape id</h1>
        <p className="mt-2 text-sm text-[var(--color-muted)]">
          The URL should look like <code>/landscape/3</code>.
        </p>
        <Link href="/" className="button button--hairline press mt-6">
          ← Back to topics
        </Link>
      </main>
    );
  }

  if (error && !landscape) {
    return (
      <main className="app-main mx-auto max-w-xl px-6 py-20 text-center">
        <p role="alert" className="text-sm text-[var(--color-danger)]">
          {error}
        </p>
        <Link href="/" className="button button--hairline press mt-6">
          ← Back to topics
        </Link>
      </main>
    );
  }

  if (!landscape) {
    return (
      <main className="app-main mx-auto max-w-xl px-6 py-20 text-center text-sm text-[var(--color-muted)]">
        Loading landscape…
      </main>
    );
  }

  const clusterCount = landscape.clusters.filter(
    (cluster) => !cluster.is_unclustered,
  ).length;

  return (
    <main className="flex h-[calc(100dvh-var(--header-height))] flex-col">
      <div className="flex items-start gap-4 border-b border-[var(--color-edge)] px-5 py-3">
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-base font-semibold">{landscape.title}</h1>
          <p className="mt-0.5 truncate text-xs text-[var(--color-muted)]">
            {landscape.papers.length} papers · {clusterCount} clusters ·
            generation {landscape.generation} · {landscape.topic}
          </p>
        </div>
        <button
          type="button"
          onClick={onExpand}
          disabled={expanding}
          className="button button--hairline press shrink-0"
        >
          {expanding ? "Growing…" : "Grow this map"}
        </button>
      </div>

      {expanding && (
        <div className="border-b border-[var(--color-edge)] px-5 py-3">
          <p className="mb-2 text-xs leading-relaxed text-[var(--color-muted)]">
            Pulling in more papers. Existing papers stay put — the layout is
            only recomputed once, at the end.
          </p>
          <StageTimeline stages={stages} borderless />
        </div>
      )}

      {landscape.summary && (
        <p className="border-b border-[var(--color-edge)] px-5 py-2.5 text-xs leading-relaxed text-[var(--color-ink-secondary)]">
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
        <div className="hidden w-[360px] shrink-0 border-l border-[var(--color-edge)] lg:block">
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

        {/* Below lg the same panel becomes a bottom sheet over the map. */}
        {selectedPaper && (
          <div className="fixed inset-x-0 bottom-0 z-30 max-h-[70dvh] overflow-y-auto border-t border-[var(--color-edge)] bg-[var(--color-panel)] shadow-[var(--shadow-pop)] lg:hidden">
            <PaperPanel
              paper={selectedPaper}
              onClose={() => setSelectedPaperId(null)}
            />
          </div>
        )}
      </div>
    </main>
  );
}
