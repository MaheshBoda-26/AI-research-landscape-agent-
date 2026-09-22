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
  onSelect: (paperId: string) => void;
};

export type PaperFlowNode = Node<PaperNodeData, "paper">;

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1).trimEnd()}…`;
}

export const PaperNode = memo(function PaperNode({
  data,
}: NodeProps<PaperFlowNode>) {
  const { paper, color, selected, pathPosition, onSelect } = data;
  // Size by relative rank within this landscape. The absolute relevance score
  // saturates near 10 for every candidate, so it cannot drive visual weight.
  const weight = (paper.relative_score ?? 5) / 10;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSelect(paper.paper_id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") onSelect(paper.paper_id);
      }}
      style={{
        borderColor: selected ? color : "var(--color-edge)",
        boxShadow: selected ? `0 0 0 2px ${color}55` : "none",
        width: 176,
        height: 58,
        scale: String(0.92 + weight * 0.12),
      }}
      className="cursor-pointer rounded-lg border bg-[var(--color-panel)] px-2.5 py-1.5 text-left transition-shadow"
    >
      <Handle type="target" position={Position.Top} className="opacity-0" />
      <div className="flex items-center gap-1.5">
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ background: color }}
        />
        <span className="text-[10px] font-semibold tabular-nums text-[var(--color-muted)]">
          #{paper.rank}
        </span>
        {pathPosition !== null && (
          <span className="rounded-sm bg-[var(--color-accent)]/15 px-1 text-[9px] font-semibold text-[var(--color-accent)]">
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
        borderColor: `${data.color}66`,
        background: `${data.color}1a`,
        color: data.color,
      }}
    >
      {data.label}
      <span className="ml-1.5 text-[10px] opacity-70">{data.count}</span>
    </div>
  );
}
