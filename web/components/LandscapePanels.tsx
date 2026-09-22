"use client";

import type { LandscapeDetail } from "@/lib/types";

function Section({
  title,
  count,
  children,
}: {
  title: string;
  count?: number;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-[var(--color-edge)] px-4 py-4">
      <h2 className="mb-2.5 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
        {title}
        {count !== undefined && count > 0 && (
          <span className="rounded-full bg-[var(--color-panel-2)] px-1.5 py-0.5 text-[10px] tabular-nums">
            {count}
          </span>
        )}
      </h2>
      {children}
    </section>
  );
}

const EMPTY_HINT =
  "Nothing here yet — this needs an LLM key, so the synthesis stage could not run.";

export function LandscapePanels({
  landscape,
  onSelectPaper,
}: {
  landscape: LandscapeDetail;
  onSelectPaper: (paperId: string) => void;
}) {
  const titleById = new Map(landscape.papers.map((p) => [p.paper_id, p.title]));
  const short = (paperId: string) => {
    const title = titleById.get(paperId);
    return title ? (title.length > 74 ? `${title.slice(0, 73)}…` : title) : paperId;
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <Section title="Clusters" count={landscape.clusters.filter((c) => !c.is_unclustered).length}>
        <ul className="space-y-2">
          {landscape.clusters.map((cluster) => (
            <li key={cluster.id} className="flex gap-2.5">
              <span
                className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: cluster.color }}
              />
              <div className="min-w-0">
                <p className="text-xs font-medium">
                  {cluster.label}
                  <span className="ml-1.5 text-[10px] font-normal text-[var(--color-muted)]">
                    {cluster.paper_count}
                  </span>
                </p>
                {cluster.description && (
                  <p className="mt-0.5 text-[11px] leading-relaxed text-[var(--color-muted)]">
                    {cluster.description}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Suggested reading path" count={landscape.reading_path.length}>
        {landscape.reading_path.length === 0 ? (
          <p className="text-[11px] text-[var(--color-muted)]">{EMPTY_HINT}</p>
        ) : (
          <ol className="space-y-2.5">
            {[...landscape.reading_path]
              .sort((a, b) => a.position - b.position)
              .map((step) => (
                <li key={step.paper_id} className="flex gap-2.5">
                  <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--color-accent)]/15 text-[10px] font-semibold text-[var(--color-accent)]">
                    {step.position}
                  </span>
                  <div className="min-w-0">
                    <button
                      type="button"
                      onClick={() => onSelectPaper(step.paper_id)}
                      className="text-left text-[11px] font-medium leading-snug hover:text-[var(--color-accent)]"
                    >
                      {short(step.paper_id)}
                    </button>
                    {step.why && (
                      <p className="mt-0.5 text-[11px] leading-relaxed text-[var(--color-muted)]">
                        {step.why}
                      </p>
                    )}
                  </div>
                </li>
              ))}
          </ol>
        )}
      </Section>

      <Section title="Tensions" count={landscape.tensions.length}>
        {landscape.tensions.length === 0 ? (
          <p className="text-[11px] text-[var(--color-muted)]">
            No paper pair in this set was found to disagree.
          </p>
        ) : (
          <ul className="space-y-3">
            {landscape.tensions.map((tension, index) => (
              <li
                key={`${tension.paper_a_id}-${tension.paper_b_id}-${index}`}
                className="rounded-lg border border-red-500/25 bg-red-500/5 px-3 py-2"
              >
                <p className="text-[11px] leading-relaxed text-[var(--color-ink)]">
                  {tension.statement}
                </p>
                <div className="mt-2 space-y-0.5">
                  {[tension.paper_a_id, tension.paper_b_id].map((paperId, i) => (
                    <button
                      key={`${paperId}-${i}`}
                      type="button"
                      onClick={() => onSelectPaper(paperId)}
                      className="block text-left text-[11px] text-[var(--color-muted)] hover:text-[var(--color-accent)]"
                    >
                      {i === 0 ? "A" : "B"} · {short(paperId)}
                    </button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Open problems" count={landscape.open_problems.length}>
        {landscape.open_problems.length === 0 ? (
          <p className="text-[11px] text-[var(--color-muted)]">{EMPTY_HINT}</p>
        ) : (
          <ul className="space-y-3">
            {landscape.open_problems.map((problem, index) => (
              <li key={index}>
                <p className="text-[11px] leading-relaxed text-[var(--color-ink)]">
                  {problem.statement}
                </p>
                {problem.why_open && (
                  <p className="mt-0.5 text-[11px] leading-relaxed text-[var(--color-muted)]">
                    {problem.why_open}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  );
}
