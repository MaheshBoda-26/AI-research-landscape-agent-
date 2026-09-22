"use client";

import { useState } from "react";
import Link from "next/link";

import type { LandscapeSummary } from "@/lib/types";

const STATUS_STYLES: Record<string, string> = {
  ready:
    "border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 text-[var(--color-accent)]",
  running:
    "border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 text-[var(--color-accent-strong)]",
  pending:
    "border-[var(--color-edge)] bg-[var(--color-panel-2)] text-[var(--color-muted)]",
  failed:
    "border-[var(--color-danger)]/30 bg-[var(--color-danger-dim)] text-[var(--color-danger)]",
};

function LandscapeRow({
  landscape,
  onDelete,
}: {
  landscape: LandscapeSummary;
  onDelete: (id: number) => void;
}) {
  // Two-step inline confirm: a one-click permanent delete is too easy to hit.
  const [confirming, setConfirming] = useState(false);

  return (
    <div className="group flex items-center gap-3 rounded-[var(--radius-lg)] border border-[var(--color-edge)] bg-[var(--color-panel)] px-4 py-3 transition-colors hover:border-[var(--color-accent)]/40">
      <Link
        href={`/landscape/${landscape.id}`}
        className="focus-ring min-w-0 flex-1 rounded-[var(--radius-sm)]"
      >
        <p className="truncate text-sm font-medium text-[var(--color-ink)]">
          {landscape.title}
        </p>
        <p className="mt-0.5 truncate text-xs text-[var(--color-muted)]">
          {landscape.paper_count} papers · {landscape.cluster_count} clusters
          {landscape.generation > 1
            ? ` · generation ${landscape.generation}`
            : ""}
        </p>
      </Link>

      <span
        className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[11px] uppercase tracking-[0.05em] ${
          STATUS_STYLES[landscape.status] ?? STATUS_STYLES.pending
        }`}
      >
        {landscape.status}
      </span>

      {confirming ? (
        <span className="flex shrink-0 items-center gap-1.5">
          <button
            type="button"
            onClick={() => onDelete(landscape.id)}
            className="cursor-pointer rounded-[var(--radius-sm)] border border-[var(--color-danger)]/40 px-2.5 py-1.5 text-xs text-[var(--color-danger)] transition-colors hover:bg-[var(--color-danger-dim)]"
          >
            Delete
          </button>
          <button
            type="button"
            onClick={() => setConfirming(false)}
            className="cursor-pointer rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs text-[var(--color-muted)] transition-colors hover:text-[var(--color-ink)]"
          >
            Keep
          </button>
        </span>
      ) : (
        <button
          type="button"
          aria-label={`Delete ${landscape.title}`}
          onClick={() => setConfirming(true)}
          className="shrink-0 cursor-pointer rounded-[var(--radius-sm)] px-2 py-1.5 text-xs text-[var(--color-muted)] transition-opacity hover:text-[var(--color-danger)] focus-visible:opacity-100 max-lg:opacity-100 lg:opacity-0 lg:group-hover:opacity-100"
        >
          Delete
        </button>
      )}
    </div>
  );
}

export function LandscapeList({
  landscapes,
  onDelete,
}: {
  landscapes: LandscapeSummary[];
  onDelete: (id: number) => void;
}) {
  if (landscapes.length === 0) {
    return (
      <p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--color-edge)] px-4 py-6 text-center text-xs text-[var(--color-muted)]">
        Nothing mapped yet. Start with a topic above.
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {landscapes.map((landscape) => (
        <li key={landscape.id}>
          <LandscapeRow landscape={landscape} onDelete={onDelete} />
        </li>
      ))}
    </ul>
  );
}
