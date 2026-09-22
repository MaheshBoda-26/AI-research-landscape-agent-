"use client";

import Link from "next/link";

import type { LandscapeSummary } from "@/lib/types";

const STATUS_STYLES: Record<string, string> = {
  ready: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  running: "border-[var(--color-accent)]/30 bg-[var(--color-accent)]/10 text-[var(--color-accent)]",
  pending: "border-[var(--color-edge)] bg-[var(--color-panel-2)] text-[var(--color-muted)]",
  failed: "border-red-500/30 bg-red-500/10 text-red-300",
};

export function LandscapeList({
  landscapes,
  onDelete,
}: {
  landscapes: LandscapeSummary[];
  onDelete: (id: number) => void;
}) {
  if (landscapes.length === 0) {
    return (
      <p className="rounded-xl border border-dashed border-[var(--color-edge)] px-4 py-6 text-center text-xs text-[var(--color-muted)]">
        Nothing mapped yet. Start with a topic above.
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {landscapes.map((landscape) => (
        <li key={landscape.id}>
          <div className="group flex items-center gap-3 rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] px-4 py-3 transition-colors hover:border-[var(--color-accent)]/40">
            <Link href={`/landscape/${landscape.id}`} className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{landscape.title}</p>
              <p className="mt-0.5 truncate text-xs text-[var(--color-muted)]">
                {landscape.paper_count} papers · {landscape.cluster_count} clusters
                {landscape.generation > 1 ? ` · generation ${landscape.generation}` : ""}
              </p>
            </Link>

            <span
              className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide ${
                STATUS_STYLES[landscape.status] ?? STATUS_STYLES.pending
              }`}
            >
              {landscape.status}
            </span>

            <button
              type="button"
              aria-label={`Delete ${landscape.title}`}
              onClick={() => onDelete(landscape.id)}
              className="shrink-0 rounded-md px-2 py-1 text-xs text-[var(--color-muted)] opacity-0 transition-opacity hover:text-red-300 focus:opacity-100 group-hover:opacity-100"
            >
              Delete
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
