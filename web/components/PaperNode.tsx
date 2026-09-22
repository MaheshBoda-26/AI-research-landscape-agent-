"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";

import type { PaperInLandscape } from "@/lib/types";

export type PaperNodeData = {
  paper: PaperInLandscape;
  color: string;
  selected: boolean;
  /** The reading order position, or null when the paper is not on the path. */
  pathPosition: number | null;
  /** Position in the map for the one-shot entrance stagger. */
  index: number;
  onSelect: (paperId: string) => void;
};

export type PaperFlowNode = Node<PaperNodeData, "paper">;

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1).trimEnd()}…`;
}

export const PaperNode = memo(function PaperNode({ data }: NodeProps<PaperFlowNode>) {
  const { paper, color, selected, pathPosition, index, onSelect } = data;
  // Size by relative rank within this landscape. The absolute relevance score
  // saturates near 10 for every candidate, so it cannot drive visual weight.
  const weight = (paper.relative_score ?? 5) / 10;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={() => onSelect(paper.paper_id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(paper.paper_id);
        }
      }}
      style={{
        // CSS vars so hover/selected states stay in one place.
        "--node-color": color,
        "--i": index,
        borderColor: selected ? "var(--node-color)" : "var(--color-edge)",
        boxShadow: selected
          ? "0 0 0 2px color-mix(in oklch, var(--node-color) 35%, transparent)"
          : "var(--shadow-1)",
        width: 180,
        height: 60,
        scale: String(0.92 + weight * 0.12),
      } as React.CSSProperties}
      className="focus-ring press cursor-pointer rounded-[var(--radius-md)] border bg-[var(--color-panel)] px-2.5 py-1.5 text-left hover:border-[var(--node-color)]"
    >
      <Handle type="target" position={Position.Top} className="opacity-0" />
      <div className="flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ background: color }}
        />
        <span className="text-[10px] font-semibold tabular-nums text-[var(--color-muted)]">
          #{paper.rank}
        </span>
        {pathPosition !== null && (
          <span className="rounded-[var(--radius-sm)] bg-[var(--color-accent)]/15 px-1.5 py-px text-[10px] font-semibold text-[var(--color-accent)]">
            step {pathPosition}
          </span>
        )}
      </div>
      <p className="mt-0.5 text-[11px] leading-tight text-[var(--color-ink)]">
        {truncate(paper.title, 66)}
      </p>
      <Handle type="source" position={Position.Bottom} className="opacity-0" />
    </div>
  );
});

export function ClusterLabelNode({
  data,
}: NodeProps<Node<{ label: string; count: number; color: string }, "cluster">>) {
  return (
    <div
      className="pointer-events-none select-none rounded-full border px-2.5 py-1 text-[11px] font-medium backdrop-blur"
      style={{
        borderColor: "color-mix(in oklch, currentColor 40%, transparent)",
        background: "color-mix(in oklch, currentColor 10%, transparent)",
        color: data.color,
      }}
    >
      {data.label}
      <span className="ml-1.5 text-[10px] opacity-70">{data.count}</span>
    </div>
  );
}
