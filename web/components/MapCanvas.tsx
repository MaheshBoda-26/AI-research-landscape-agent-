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

import { ClusterLabelNode, PaperNode, type PaperFlowNode } from "@/components/PaperNode";
import { EDGE_LABELS, type EdgeKind, type LandscapeDetail } from "@/lib/types";

/** UMAP output is roughly unit-scale; this maps it onto readable pixel space. */
const SCALE = 460;
const UNCLUSTERED_COLOR = "#7b88a3";

const EDGE_COLORS: Record<EdgeKind, string> = {
  builds_on: "#5eead4",
  improves: "#6ea8fe",
  compares_against: "#a78bfa",
  contradicts: "#f87171",
  shares_method: "#9ca3af",
  applies: "#fbbf24",
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
    const paperNodes: FlowNode[] = landscape.papers.map((paper) => {
      const cluster =
        paper.cluster_id !== null ? clusterById.get(paper.cluster_id) : undefined;
      return {
        id: paper.paper_id,
        type: "paper" as const,
        position: { x: paper.x * SCALE, y: paper.y * SCALE },
        data: {
          paper,
          color: cluster?.is_unclustered
            ? UNCLUSTERED_COLOR
            : (cluster?.color ?? UNCLUSTERED_COLOR),
          selected: paper.paper_id === selectedPaperId,
          pathPosition: pathPositions.get(paper.paper_id) ?? null,
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
        color: cluster.is_unclustered ? UNCLUSTERED_COLOR : cluster.color,
      },
      selectable: false,
      draggable: false,
      zIndex: 10,
    }));

    return [...labelNodes, ...paperNodes];
  }, [landscape.papers, landscape.clusters, clusterById, pathPositions, selectedPaperId, onSelectPaper]);

  const edges = useMemo<FlowEdge[]>(() => {
    if (!showEdges) return [];
    return landscape.edges.map((edge, index) => ({
      id: `edge-${index}-${edge.src_paper_id}-${edge.dst_paper_id}`,
      source: edge.src_paper_id,
      target: edge.dst_paper_id,
      label: edge.rationale ? undefined : EDGE_LABELS[edge.kind],
      style: {
        stroke: EDGE_COLORS[edge.kind] ?? "#64748b",
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
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e2942" />
        <Controls showInteractive={false} />
        <MiniMap
          pannable
          zoomable
          nodeColor={(node) =>
            node.type === "cluster"
              ? "transparent"
              : ((node.data as { color?: string })?.color ?? "#64748b")
          }
          maskColor="rgb(11 16 32 / 70%)"
          className="!bg-[var(--color-panel)]"
        />
      </ReactFlow>

      <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2">
        <div className="pointer-events-auto flex items-center gap-2 rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)]/90 px-3 py-1.5 text-[11px] backdrop-blur">
          <span className="text-[var(--color-muted)]">
            {landscape.papers.length} papers · {landscape.clusters.filter((c) => !c.is_unclustered).length} clusters
          </span>
          <button
            type="button"
            onClick={() => setShowEdges((value) => !value)}
            className="rounded border border-[var(--color-edge)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)]"
          >
            {showEdges ? "hide links" : "show links"}
          </button>
        </div>
      </div>
    </div>
  );
}
