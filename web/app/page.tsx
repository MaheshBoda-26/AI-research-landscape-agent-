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
        cause instanceof Error
          ? `Could not reach the API: ${cause.message}`
          : "Could not reach the API.",
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
      <main className="app-main">
        <RunView
          topic={topic}
          active
          onFinished={(id) => router.push(`/landscape/${id}`)}
          onCancel={() => setTopic(null)}
        />
      </main>
    );
  }

  return (
    <main className="app-main mx-auto w-full max-w-3xl px-6 py-16">
      <div className="mb-12 text-center">
        <h1>What do you want to understand?</h1>
        <p className="mx-auto mt-4 max-w-xl text-[15px] leading-relaxed text-[var(--color-ink-secondary)]">
          Enter an ML topic in plain English. arXiv supplies the candidates, the
          model reads and ranks them, and you get a map of the area rather than
          a list of papers.
        </p>
      </div>

      <TopicForm onSubmit={setTopic} />

      {health && !health.llm_configured && (
        <div className="mt-10 rounded-[var(--radius-lg)] border border-[var(--color-warning)]/30 bg-[var(--color-warning-dim)] px-4 py-3">
          <p className="text-xs font-medium text-[var(--color-warning)]">
            No LLM API key is configured.
          </p>
          <p className="mt-1 text-xs leading-relaxed text-[var(--color-ink-secondary)]">
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
        <div
          role="alert"
          className="mt-10 rounded-[var(--radius-lg)] border border-[var(--color-danger)]/30 bg-[var(--color-danger-dim)] px-4 py-3 text-xs leading-relaxed text-[var(--color-danger)]"
        >
          {error}
        </div>
      )}

      <section className="mt-14">
        <h2 className="mb-3 text-xs font-medium uppercase tracking-[0.05em] text-[var(--color-muted)]">
          Your landscapes
        </h2>
        <LandscapeList landscapes={landscapes} onDelete={onDelete} />
      </section>
    </main>
  );
}
