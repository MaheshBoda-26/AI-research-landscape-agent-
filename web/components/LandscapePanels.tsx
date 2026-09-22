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
    <section className="border-t border-[var(--color-edge)] px-4 py-4 first:border-t-0">
      <h2 className="mb-2.5 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.05em] text-[var(--color-muted)]">
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
      <Section
        title="Clusters"
        count={landscape.clusters.filter((c) => !c.is_unclustered).length}
      >
        <ul className="space-y-2.5">
          {landscape.clusters.map((cluster) => (
            <li key={cluster.id} className="flex gap-2.5">
              <span
                aria-hidden="true"
                className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: cluster.color }}
              />
              <div className="min-w-0">
                <p className="text-xs font-medium text-[var(--color-ink)]">
                  {cluster.label}
                  <span className="ml-1.5 text-[11px] font-normal tabular-nums text-[var(--color-muted)]">
                    {cluster.paper_count}
                  </span>
                </p>
                {cluster.description && (
                  <p className="mt-0.5 text-xs leading-relaxed text-[var(--color-ink-secondary)]">
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
          <p className="text-xs text-[var(--color-muted)]">{EMPTY_HINT}</p>
        ) : (
          <ol className="space-y-2.5">
            {[...landscape.reading_path]
              .sort((a, b) => a.position - b.position)
              .map((step) => (
                <li key={step.paper_id} className="flex gap-2.5">
                  <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--color-accent)]/15 text-[10px] font-semibold tabular-nums text-[var(--color-accent)]">
                    {step.position}
                  </span>
                  <div className="min-w-0">
                    <button
                      type="button"
                      onClick={() => onSelectPaper(step.paper_id)}
                      className="focus-ring cursor-pointer rounded-[var(--radius-sm)] text-left text-xs font-medium leading-snug text-[var(--color-ink)] transition-colors hover:text-[var(--color-accent)]"
                    >
                      {short(step.paper_id)}
                    </button>
                    {step.why && (
                      <p className="mt-0.5 text-xs leading-relaxed text-[var(--color-ink-secondary)]">
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
          <p className="text-xs text-[var(--color-muted)]">
            No paper pair in this set was found to disagree.
          </p>
        ) : (
          <ul className="space-y-3">
            {landscape.tensions.map((tension, index) => (
              <li
                key={`${tension.paper_a_id}-${tension.paper_b_id}-${index}`}
                className="rounded-[var(--radius-md)] border border-[var(--color-danger)]/25 bg-[var(--color-danger-dim)] px-3 py-2"
              >
                <p className="text-xs leading-relaxed text-[var(--color-ink)]">
                  {tension.statement}
                </p>
                <div className="mt-2 space-y-0.5">
                  {[tension.paper_a_id, tension.paper_b_id].map((paperId, i) => (
                    <button
                      key={`${paperId}-${i}`}
                      type="button"
                      onClick={() => onSelectPaper(paperId)}
                      className="focus-ring block cursor-pointer rounded-[var(--radius-sm)] py-0.5 text-left text-xs text-[var(--color-ink-secondary)] transition-colors hover:text-[var(--color-accent)]"
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
          <p className="text-xs text-[var(--color-muted)]">{EMPTY_HINT}</p>
        ) : (
          <ul className="space-y-3 pb-6">
            {landscape.open_problems.map((problem, index) => (
              <li key={index}>
                <p className="text-xs leading-relaxed text-[var(--color-ink)]">
                  {problem.statement}
                </p>
                {problem.why_open && (
                  <p className="mt-0.5 text-xs leading-relaxed text-[var(--color-ink-secondary)]">
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
