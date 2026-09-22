"use client";

import { useCallback, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  ClusterLabelNode,
  PaperNode,
  type PaperFlowNode,
} from "@/components/PaperNode";
import { EDGE_LABELS, type EdgeKind, type LandscapeDetail } from "@/lib/types";

/** UMAP output is roughly unit-scale; this maps it onto readable pixel space. */
const SCALE = 460;

/**
 * Relationship colors. All drawn from the map hue family plus the two semantic
 * exceptions (danger = contradiction, muted = shared method), so the canvas
 * never reads as a rainbow. Contradiction keeps the dashed treatment.
 */
const EDGE_COLORS: Record<EdgeKind, string> = {
  builds_on: "var(--color-hue-1)",
  improves: "var(--color-hue-2)",
  compares_against: "var(--color-hue-3)",
  contradicts: "var(--color-danger)",
  shares_method: "var(--color-muted)",
  applies: "var(--color-hue-4)",
};

const nodeTypes = { paper: PaperNode, cluster: ClusterLabelNode };

export function MapCanvas({
  landscape,
  selectedPaperId,
  onSelectPaper,
}: {
  landscape: LandscapeDetail;
  selectedPaperId: string | null;
  onSelectPaper: (paperId: string) => void;
}) {
  const [showEdges, setShowEdges] = useState(true);

  const clusterById = useMemo(
    () => new Map(landscape.clusters.map((cluster) => [cluster.id, cluster])),
    [landscape.clusters],
  );

  const pathPositions = useMemo(
    () =>
      new Map(
        landscape.reading_path.map((step) => [step.paper_id, step.position]),
      ),
    [landscape.reading_path],
  );

  const nodes = useMemo<FlowNode[]>(() => {
    const paperNodes: FlowNode[] = landscape.papers.map((paper, index) => {
      const cluster =
        paper.cluster_id !== null ? clusterById.get(paper.cluster_id) : undefined;
      return {
        id: paper.paper_id,
        type: "paper" as const,
        position: { x: paper.x * SCALE, y: paper.y * SCALE },
        data: {
          paper,
          color: cluster?.color ?? "var(--color-unclustered)",
          selected: paper.paper_id === selectedPaperId,
          pathPosition: pathPositions.get(paper.paper_id) ?? null,
          // Stagger index for the one-shot entrance animation.
          index,
          onSelect: onSelectPaper,
        },
      };
    });

    // Cluster labels sit above the paper nodes and are never interactive, so
    // they read as region annotations rather than as something clickable.
    const labelNodes: FlowNode[] = landscape.clusters.map((cluster) => ({
      id: `cluster-${cluster.id}`,
      type: "cluster",
      position: { x: cluster.x * SCALE, y: cluster.y * SCALE - 96 },
      data: {
        label: cluster.label,
        count: cluster.paper_count,
        color: cluster.is_unclustered
          ? "var(--color-unclustered)"
          : cluster.color,
      },
      selectable: false,
      draggable: false,
      zIndex: 10,
    }));

    return [...labelNodes, ...paperNodes];
  }, [
    landscape.papers,
    landscape.clusters,
    clusterById,
    pathPositions,
    selectedPaperId,
    onSelectPaper,
  ]);

  const edges = useMemo<FlowEdge[]>(() => {
    if (!showEdges) return [];
    return landscape.edges.map((edge, index) => ({
      id: `edge-${index}-${edge.src_paper_id}-${edge.dst_paper_id}`,
      source: edge.src_paper_id,
      target: edge.dst_paper_id,
      label: edge.rationale ? undefined : EDGE_LABELS[edge.kind],
      style: {
        stroke: EDGE_COLORS[edge.kind] ?? "var(--color-muted)",
        strokeWidth: 1 + edge.weight,
        // A contradiction is the one relationship worth spotting from a
        // distance, so it is the only dashed edge kind.
        strokeDasharray: edge.kind === "contradicts" ? "5 4" : undefined,
      },
      animated: edge.kind === "contradicts",
    }));
  }, [landscape.edges, showEdges]);

  const onNodeClick = useCallback<NodeMouseHandler>(
    (_event, node) => {
      if (node.type === "paper") onSelectPaper(node.id);
    },
    [onSelectPaper],
  );

  return (
    <div className="relative h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.25, maxZoom: 1.4 }}
        minZoom={0.12}
        proOptions={{ hideAttribution: false }}
        className="bg-[var(--color-canvas)]"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={22}
          size={1}
          color="var(--color-dots)"
        />
        <Controls showInteractive={false} />
        <MiniMap
          pannable
          zoomable
          nodeColor={(node) =>
            node.type === "cluster"
              ? "transparent"
              : ((node.data as { color?: string })?.color ?? "var(--color-muted)")
          }
          maskColor="oklch(0.145 0.012 180 / 0.7)"
        />
      </ReactFlow>

      <div className="pointer-events-none absolute left-3 top-3">
        <div className="pointer-events-auto flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-edge)] bg-[var(--color-panel)]/90 px-3 py-1.5 text-xs text-[var(--color-muted)] backdrop-blur">
          <button
            type="button"
            aria-pressed={showEdges}
            onClick={() => setShowEdges((value) => !value)}
            className="focus-ring cursor-pointer rounded-[var(--radius-sm)] px-1 py-0.5 text-[var(--color-ink-secondary)] transition-colors hover:text-[var(--color-ink)]"
          >
            {showEdges ? "Hide links" : "Show links"}
          </button>
          <span aria-hidden="true" className="text-[var(--color-edge-strong)]">
            |
          </span>
          <span className="tabular-nums">{landscape.papers.length} papers</span>
        </div>
      </div>
    </div>
  );
}
