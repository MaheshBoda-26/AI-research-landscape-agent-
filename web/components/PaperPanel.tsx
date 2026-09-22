"use client";

import type { PaperExtraction, PaperInLandscape } from "@/lib/types";

const FIELD_LABELS: Array<[keyof PaperExtraction & string, string]> = [
  ["problem", "Problem"],
  ["method", "Method"],
  ["results", "Results"],
  ["contribution", "Contribution"],
  ["limitations", "Limitations"],
];

function ScoreRow({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-[11px] text-[var(--color-muted)]">{label}</span>
      <span className="text-right">
        <span className="text-xs font-medium tabular-nums">{value}</span>
        <span className="ml-2 text-[10px] text-[var(--color-muted)]">{hint}</span>
      </span>
    </div>
  );
}

function EvidenceQuote({ quote }: { quote: string }) {
  return (
    <blockquote className="mt-1 border-l-2 border-[var(--color-accent)]/40 pl-2 text-[11px] italic text-[var(--color-muted)]">
      “{quote}”
    </blockquote>
  );
}

export function PaperPanel({
  paper,
  onClose,
}: {
  paper: PaperInLandscape;
  onClose: () => void;
}) {
  const extraction = paper.extraction;

  return (
    <aside className="flex h-full w-full flex-col overflow-hidden border-l border-[var(--color-edge)] bg-[var(--color-panel)]">
      <div className="flex items-start gap-3 border-b border-[var(--color-edge)] px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
            Rank #{paper.rank} · {paper.primary_category || "uncategorised"}
          </p>
          <h2 className="mt-0.5 text-sm font-semibold leading-snug">
            {paper.title}
          </h2>
          <p className="mt-1 truncate text-[11px] text-[var(--color-muted)]">
            {paper.authors.slice(0, 4).join(", ")}
            {paper.authors.length > 4 ? " et al." : ""}
            {paper.published ? ` · ${paper.published.slice(0, 10)}` : ""}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="shrink-0 rounded-md px-2 py-1 text-xs text-[var(--color-muted)] hover:text-[var(--color-ink)]"
        >
          ✕
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <div className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel-2)]/50 px-3 py-2">
          <ScoreRow
            label="Relevance"
            value={paper.relevance_score.toFixed(2)}
            hint="absolute 0-10"
          />
          {paper.cross_encoder_logit !== null && (
            <ScoreRow
              label="Cross-encoder logit"
              value={paper.cross_encoder_logit.toFixed(2)}
              hint="raw, resolves order"
            />
          )}
          {paper.relative_score !== null && (
            <ScoreRow
              label="Within landscape"
              value={paper.relative_score.toFixed(1)}
              hint="percentile"
            />
          )}
          <ScoreRow label="Scored by" value={paper.rerank_source} hint="" />
        </div>

        <p className="mt-2 text-[10px] leading-relaxed text-[var(--color-muted)]">
          Relevance saturates near 10 for every arXiv candidate, so the logit —
          not the 0-10 score — is what orders this list.
        </p>

        <div className="mt-4 flex gap-2">
          <a
            href={paper.abs_url}
            target="_blank"
            rel="noreferrer"
            className="rounded-md bg-[var(--color-accent)]/15 px-2.5 py-1 text-[11px] font-medium text-[var(--color-accent)] hover:bg-[var(--color-accent)]/25"
          >
            arXiv abstract
          </a>
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-[var(--color-edge)] px-2.5 py-1 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]"
            >
              PDF
            </a>
          )}
        </div>

        <section className="mt-5">
          <h3 className="text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
            Reading
          </h3>
          {extraction ? (
            <dl className="mt-2 space-y-3">
              {FIELD_LABELS.map(([key, label]) => {
                const value = extraction[key] as string | null;
                if (!value) return null;
                const quote = extraction.evidence?.[key];
                return (
                  <div key={key}>
                    <dt className="text-[11px] font-semibold text-[var(--color-ink)]">
                      {label}
                    </dt>
                    <dd className="text-xs leading-relaxed text-[var(--color-muted)]">
                      {value}
                      {quote && <EvidenceQuote quote={quote} />}
                    </dd>
                  </div>
                );
              })}
            </dl>
          ) : (
            <p className="mt-2 text-xs text-[var(--color-muted)]">
              No structured extraction for this paper. That happens when no LLM
              key is configured, or when the model's answer failed the verbatim
              evidence check.
            </p>
          )}

          {extraction && (extraction.datasets.length > 0 || extraction.metrics.length > 0) && (
            <div className="mt-3 flex flex-wrap gap-1">
              {[...extraction.datasets, ...extraction.metrics].map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-[var(--color-edge)] px-2 py-0.5 text-[10px] text-[var(--color-muted)]"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </section>

        <section className="mt-5">
          <h3 className="text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
            Abstract
          </h3>
          <p className="mt-2 text-[11px] leading-relaxed text-[var(--color-muted)]">
            {paper.abstract}
          </p>
        </section>
      </div>
    </aside>
  );
}
