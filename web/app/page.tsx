"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { LandscapeList } from "@/components/LandscapeList";
import { RunView } from "@/components/RunView";
import { TopicForm } from "@/components/TopicForm";
import { deleteLandscape, getHealth, listLandscapes } from "@/lib/api";
import type { HealthOut, LandscapeSummary } from "@/lib/types";

export default function HomePage() {
  const router = useRouter();
  const [health, setHealth] = useState<HealthOut | null>(null);
  const [landscapes, setLandscapes] = useState<LandscapeSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [topic, setTopic] = useState<string | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const [nextHealth, nextLandscapes] = await Promise.all([
        getHealth(signal),
        listLandscapes(signal),
      ]);
      setHealth(nextHealth);
      setLandscapes(nextLandscapes);
      setError(null);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(
        cause instanceof Error ? cause.message : "Could not reach the API.",
      );
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const onDelete = useCallback(
    async (id: number) => {
      await deleteLandscape(id);
      void refresh();
    },
    [refresh],
  );

  if (topic) {
    return (
      <RunView
        topic={topic}
        active
        onFinished={(id) => router.push(`/landscape/${id}`)}
      />
    );
  }

  return (
    <main className="mx-auto w-full max-w-3xl px-6 py-14">
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-semibold tracking-tight">
          What do you want to understand?
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-sm text-[var(--color-muted)]">
          Enter an ML topic in plain English. arXiv supplies the candidates, the
          model reads and ranks them, and you get a map of the area rather than a
          list of papers.
        </p>
      </div>

      <TopicForm onSubmit={setTopic} />

      {health && !health.llm_configured && (
        <div className="mt-8 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3">
          <p className="text-xs font-medium text-amber-300">
            No LLM API key is configured.
          </p>
          <p className="mt-1 text-xs text-[var(--color-muted)]">
            Retrieval, reranking, and the layout still work — you will get real
            papers positioned by real embedding structure. But without a key
            there are no extractions, no cluster names, and no synthesis. Set{" "}
            <code className="text-[var(--color-ink)]">NVIDIA_API_KEY</code> or{" "}
            <code className="text-[var(--color-ink)]">OPENROUTER_API_KEY</code> in
            the backend <code className="text-[var(--color-ink)]">.env</code>.
          </p>
        </div>
      )}

      {error && (
        <div className="mt-8 rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-300">
          {error}
        </div>
      )}

      <section className="mt-12">
        <h2 className="mb-3 text-xs uppercase tracking-wider text-[var(--color-muted)]">
          Your landscapes
        </h2>
        <LandscapeList landscapes={landscapes} onDelete={onDelete} />
      </section>
    </main>
  );
}
