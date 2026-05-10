"use client";

import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { createPortal } from "react-dom";
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  Handle,
  Position,
  NodeProps,
  NodeTypes,
  MarkerType,
  ReactFlowInstance,
} from "reactflow";
import "reactflow/dist/style.css";
import * as d3 from "d3-force";
import { api } from "@/lib/api";
import type { GraphResponse, GraphNode, GraphEdge } from "@/types";
import { useNamespaceStore } from "@/store/namespace";
import { useJobsStore } from "@/store/jobs";
import {
  Loader2Icon, XIcon, BookmarkIcon, RefreshCwIcon,
  NetworkIcon, SearchIcon, ZapIcon, ExternalLinkIcon,
  ChevronDownIcon, ChevronRightIcon, EyeOffIcon, FolderIcon, CheckIcon,
  Trash2Icon,
} from "lucide-react";
import type { BookmarkFolder, Bookmark } from "@/types";

// ─── Palette ───────────────────────────────────────────────────────────────────

const P = {
  TOPIC:    { ring: "#6366f1", glow: "rgba(99,102,241,0.4)",  bg: "rgba(15,14,39,0.97)",  text: "#a5b4fc", badge: "#312e81" },
  SUBTOPIC: { ring: "#8b5cf6", glow: "rgba(139,92,246,0.35)", bg: "rgba(26,14,51,0.97)",  text: "#c4b5fd", badge: "#4c1d95" },
  CONCEPT:  { ring: "#0d9488", glow: "rgba(13,148,136,0.3)",  bg: "rgba(4,31,30,0.97)",   text: "#5eead4", badge: "#134e4a" },
  METHOD:   { ring: "#f59e0b", glow: "rgba(245,158,11,0.3)",  bg: "rgba(28,15,0,0.97)",   text: "#fcd34d", badge: "#78350f" },
  PAPER:    { ring: "#374151", glow: "rgba(55,65,81,0.25)",   bg: "rgba(8,11,18,0.98)",   text: "#d1d5db", badge: "#1f2937" },
} as const;

const RADIUS: Record<string, number> = { TOPIC: 72, SUBTOPIC: 54, CONCEPT: 38, METHOD: 38 };

// ─── Cluster node (concepts / methods / topics) ────────────────────────────────

function ClusterNode({ data, selected }: NodeProps) {
  const type      = data.type as keyof typeof P;
  const isCluster = !!(data.isCluster as boolean);
  const isSubject = !!(data.isSubject as boolean);
  const p = P[type] ?? P.CONCEPT;
  const r = (data.r as number) || RADIUS[type] || 38;
  const isExpanded = !!data.expanded;
  const isMatched  = !!data.matched;
  const childCount = (data.childCount as number) || 0;
  const label = (data.label as string) || "";
  const short = label.length > 24 ? label.slice(0, 23) + "…" : label;
  // Subject TOPIC nodes (root of hierarchy) get a special badge to distinguish
  // them from domain TOPIC nodes (Artificial Intelligence, Computer Vision, etc.)
  const badge = isSubject ? "SUBJECT" : isCluster ? "CLUSTER" : type;

  return (
    <>
      <Handle type="target" position={Position.Top}    style={{ opacity: 0, pointerEvents: "none" }} />
      {/* search highlight ring */}
      {isMatched && (
        <div style={{
          position: "absolute", inset: -10, borderRadius: "50%",
          border: "2.5px solid #fbbf24",
          boxShadow: "0 0 18px rgba(251,191,36,0.7)",
          animation: "pulse 1.5s ease-in-out infinite",
          pointerEvents: "none",
        }} />
      )}
      {/* expanded pulse */}
      {isExpanded && !isMatched && (
        <div style={{
          position: "absolute", inset: -6, borderRadius: "50%",
          border: `1.5px solid ${p.ring}`, opacity: 0.25,
          animation: "pulse 2.5s ease-in-out infinite", pointerEvents: "none",
        }} />
      )}
      {/* ambient glow */}
      <div style={{
        position: "absolute", inset: -18, borderRadius: "50%",
        background: `radial-gradient(circle, ${isMatched ? "rgba(251,191,36,0.15)" : p.glow} 0%, transparent 70%)`,
        pointerEvents: "none", opacity: selected ? 1 : 0.7, transition: "opacity 0.2s",
      }} />
      {/* main circle */}
      <div style={{
        width: r * 2, height: r * 2, borderRadius: "50%",
        background: p.bg,
        border: `2px solid ${isMatched ? "#fbbf24" : selected ? "#fff" : p.ring}`,
        boxShadow: selected
          ? `0 0 0 3px rgba(255,255,255,0.12), 0 0 24px ${p.glow}`
          : isMatched
            ? "0 0 0 2px rgba(251,191,36,0.3), 0 0 20px rgba(251,191,36,0.3)"
            : `0 0 16px ${p.glow}, inset 0 1px 0 rgba(255,255,255,0.06)`,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        cursor: "pointer", textAlign: "center", padding: "8px",
        transition: "border-color 0.2s, box-shadow 0.2s",
        backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
        userSelect: "none", position: "relative",
      }}>
        <div style={{
          position: "absolute", top: 5,
          background: isMatched ? "rgba(251,191,36,0.2)" : p.badge,
          borderRadius: 5, padding: "1px 5px",
          fontSize: "6.5px", fontWeight: 800, letterSpacing: "0.07em",
          color: isMatched ? "#fbbf24" : p.text, textTransform: "uppercase",
        }}>{badge}</div>
        <p style={{
          color: isMatched ? "#fde68a" : p.text,
          fontSize: type === "TOPIC" ? "11px" : "10px",
          fontWeight: type === "TOPIC" ? 700 : 600,
          lineHeight: 1.25, marginTop: "10px", maxWidth: "88%",
        }}>{short}</p>
        {childCount > 0 && (
          <div style={{
            position: "absolute", bottom: 5,
            display: "flex", alignItems: "center", gap: 2,
          }}>
            {isExpanded
              ? <ChevronDownIcon size={8} color={p.ring} />
              : <ChevronRightIcon size={8} color={p.ring} />}
            <span style={{ fontSize: "7.5px", color: p.ring, fontWeight: 700 }}>{childCount}</span>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, pointerEvents: "none" }} />
    </>
  );
}

// ─── Paper node ────────────────────────────────────────────────────────────────

function PaperNode({ data, selected }: NodeProps) {
  const isBookmarked = !!data.isBookmarked;
  const isExpanded   = !!data.expanded;
  const isMatched    = !!data.matched;
  const label        = (data.label as string) || "";
  const ns           = data.namespace_key as string | null;
  const childCount   = (data.childCount as number) || 0;
  const description  = (data.description as string) || "";  // TL;DR or abstract snippet

  const borderColor = isMatched ? "#fbbf24" : selected ? "#fff" : isBookmarked ? "rgba(34,197,94,0.55)" : "rgba(55,65,81,0.7)";
  const leftColor   = isMatched ? "#fbbf24" : isBookmarked ? "#22c55e" : "rgba(55,65,81,0.8)";
  // Native tooltip: full title + TL;DR so users can read context without clicking
  const tooltip = description ? `${label}\n\n${description}` : label;

  return (
    <>
      <Handle type="target" position={Position.Top}    style={{ opacity: 0, pointerEvents: "none" }} />
      {isMatched && (
        <div style={{
          position: "absolute", inset: -8, borderRadius: "14px",
          border: "2px solid #fbbf24",
          boxShadow: "0 0 20px rgba(251,191,36,0.5)",
          animation: "pulse 1.5s ease-in-out infinite",
          pointerEvents: "none",
        }} />
      )}
      <div title={tooltip} style={{
        background: isBookmarked ? "rgba(5,20,10,0.98)" : "rgba(8,11,18,0.98)",
        border: `1.5px solid ${borderColor}`,
        borderLeft: `3.5px solid ${leftColor}`,
        borderRadius: "11px", padding: "9px 11px", width: "152px",
        boxShadow: selected
          ? "0 0 0 2px rgba(255,255,255,0.12)"
          : isMatched
            ? "0 4px 20px rgba(251,191,36,0.2)"
            : isBookmarked
              ? "0 4px 20px rgba(34,197,94,0.1), 0 1px 0 rgba(255,255,255,0.04)"
              : "0 4px 16px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03)",
        cursor: "pointer", backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)", userSelect: "none",
        transition: "border-color 0.2s, box-shadow 0.2s",
      }}>
        <p style={{
          color: isMatched ? "#fde68a" : isBookmarked ? "#86efac" : "#d1d5db",
          fontSize: "10px", fontWeight: 600, lineHeight: 1.35,
          wordBreak: "break-word", overflow: "hidden",
          display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical" as const,
          marginBottom: "5px",
        }}>{label}</p>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {isBookmarked && <BookmarkIcon size={8} color="#22c55e" fill="#22c55e" />}
            {ns && <span style={{ fontSize: "8px", color: "rgba(107,114,128,0.7)", fontWeight: 500 }}>{ns}</span>}
          </div>
          {childCount > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
              {isExpanded
                ? <ChevronDownIcon size={8} color="#6366f1" />
                : <ChevronRightIcon size={8} color="#6366f1" />}
              <span style={{ fontSize: "7.5px", color: "#6366f1", fontWeight: 700 }}>{childCount}</span>
            </div>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, pointerEvents: "none" }} />
    </>
  );
}

const nodeTypes: NodeTypes = { clusterNode: ClusterNode, paperNode: PaperNode };

// ─── Force layout ──────────────────────────────────────────────────────────────

interface SimNode extends d3.SimulationNodeDatum {
  id: string; type: string; r: number;
  isSubject: boolean; // level 0: TOPIC with no TOPIC parent (e.g. "Computer Science")
  isArea: boolean;    // level 3: CONCEPT with SUBTOPIC parent
  isSubArea: boolean; // level 4: CONCEPT with area CONCEPT parent (middle tier)
  isCluster: boolean; // level 5: CONCEPT with sub-area CONCEPT parent (leaf cluster)
}

// Y-anchor per hierarchy level — 9 tiers (added SUBAREA between AREA and CLUSTER)
const LEVEL_Y: Record<string, number> = {
  SUBJECT:  -1400,  // subject root (e.g. "Computer Science")
  TOPIC:    -1050,  // domain topic (e.g. "Artificial Intelligence")
  SUBTOPIC:  -650,  // arXiv namespace (e.g. "cs.AI")
  AREA:      -280,  // LLM research area
  SUBAREA:   -50,   // LLM sub-area (middle tier when 3-level hierarchy exists)
  CLUSTER:    180,  // LLM thematic cluster
  PAPER:      620,
  CONCEPT:   1060,
  METHOD:    1060,
};

function _nodeTierY(d: SimNode): number {
  if (d.isSubject) return LEVEL_Y.SUBJECT;
  if (d.isArea)    return LEVEL_Y.AREA;
  if (d.isSubArea) return LEVEL_Y.SUBAREA;
  if (d.isCluster) return LEVEL_Y.CLUSTER;
  return LEVEL_Y[d.type] ?? 0;
}

function runForce(
  gNodes: GraphNode[], gEdges: GraphEdge[],
  visibleIds: Set<string>,
  areaConceptIds: Set<string>,
  clusterConceptIds: Set<string>,  // = belowAreaConcepts (sub-areas + true clusters combined)
  subjectTopicIds: Set<string>,
  subAreaConceptIds: Set<string> = new Set(),  // just the sub-area tier, for Y separation
): Map<string, {x:number;y:number}> {
  const vis = gNodes.filter(n => visibleIds.has(n.id));
  if (!vis.length) return new Map();

  const simNodes: SimNode[] = vis.map(n => {
    const isSubject = subjectTopicIds.has(n.id);
    const isArea    = areaConceptIds.has(n.id);
    const isSubArea = subAreaConceptIds.has(n.id);
    // True cluster = in clusterConceptIds but NOT a sub-area (sub-areas also appear in
    // belowAreaConcepts which is passed as clusterConceptIds for internal gating logic)
    const isCluster = clusterConceptIds.has(n.id) && !isSubArea;
    const r = n.type === "PAPER" ? 65
      : isSubject ? 72
      : isArea    ? 54
      : isSubArea ? 48
      : isCluster ? 44
      : (RADIUS[n.type] ?? 38);
    const node: SimNode = { id: n.id, type: n.type, r, isSubject, isArea, isSubArea, isCluster };
    // Pin TOPIC/SUBTOPIC Y so they never drift into the cluster zone
    if (n.type === "TOPIC" || n.type === "SUBTOPIC") {
      node.fy = _nodeTierY(node);
    }
    return node;
  });

  // ── Hierarchical pre-positioning ─────────────────────────────────────────
  // Place children near their parent's x position. Without this, all nodes at
  // the same tier are spread evenly across the canvas, forcing the simulation
  // to do a lot of work to drag children back near their parents — and with
  // many siblings the result can leave some children far from the parent,
  // making them look "stray" even though they're properly connected.
  const tierKey = (n: SimNode) =>
    n.isSubject ? "SUBJECT" : n.isArea ? "AREA" : n.isSubArea ? "SUBAREA" : n.isCluster ? "CLUSTER" : n.type;
  const tierOrder = ["SUBJECT", "TOPIC", "SUBTOPIC", "AREA", "SUBAREA", "CLUSTER", "PAPER", "CONCEPT", "METHOD"];
  const tierGroups = new Map<string, SimNode[]>();
  for (const n of simNodes) {
    const k = tierKey(n);
    if (!tierGroups.has(k)) tierGroups.set(k, []);
    tierGroups.get(k)!.push(n);
  }
  // Build a parent lookup: for each node, find a visible parent (if any).
  const visibleNodeIds = new Set(simNodes.map(n => n.id));
  const parentOf: Record<string, string | undefined> = {};
  for (const e of gEdges) {
    if (visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target) && !parentOf[e.target]) {
      parentOf[e.target] = e.source;
    }
  }
  const placed = new Set<string>();
  for (const tier of tierOrder) {
    const group = tierGroups.get(tier);
    if (!group) continue;
    const ty = _nodeTierY(group[0]);
    if (tier === "SUBJECT" || (tier === "TOPIC" && !tierGroups.has("SUBJECT"))) {
      // Roots: spread evenly horizontally
      const spread = Math.max(group.length * 600, 400);
      group.forEach((n, i) => {
        n.x = (i / Math.max(1, group.length - 1) - 0.5) * spread + (Math.random() - 0.5) * 40;
        n.y = ty + (Math.random() - 0.5) * 30;
        placed.add(n.id);
      });
    } else {
      // Children: cluster around their parent's x
      const childrenByParent = new Map<string, SimNode[]>();
      const orphans: SimNode[] = [];
      for (const n of group) {
        const pid = parentOf[n.id];
        const parentNode = pid ? simNodes.find(s => s.id === pid) : undefined;
        if (parentNode && placed.has(pid!)) {
          if (!childrenByParent.has(pid!)) childrenByParent.set(pid!, []);
          childrenByParent.get(pid!)!.push(n);
        } else {
          orphans.push(n);
        }
      }
      // Place each parent's children in a fan around its x
      for (const [pid, kids] of childrenByParent) {
        const parent = simNodes.find(s => s.id === pid)!;
        const fanSpread = Math.max(kids.length * 200, 240);
        kids.forEach((n, i) => {
          const t = kids.length === 1 ? 0 : (i / (kids.length - 1)) - 0.5;
          n.x = (parent.x ?? 0) + t * fanSpread + (Math.random() - 0.5) * 30;
          n.y = ty + (Math.random() - 0.5) * 30;
          placed.add(n.id);
        });
      }
      // Orphans (no placed parent): spread along the tier
      const orphanSpread = Math.max(orphans.length * 220, 400);
      orphans.forEach((n, i) => {
        n.x = (i / Math.max(1, orphans.length - 1) - 0.5) * orphanSpread + (Math.random() - 0.5) * 40;
        n.y = ty + (Math.random() - 0.5) * 30;
        placed.add(n.id);
      });
    }
  }

  const idSet = new Set(visibleIds);
  const links = gEdges
    .filter(e => idSet.has(e.source) && idSet.has(e.target))
    .map(e => ({ source: e.source, target: e.target }));

  const repel = (d: SimNode) => {
    if (d.isSubject)           return -3200;
    if (d.type === "TOPIC")    return -2600;
    if (d.type === "SUBTOPIC") return -2400;
    if (d.isArea)              return -1100;
    if (d.isSubArea)           return -900;
    if (d.isCluster)           return -800;
    if (d.type === "PAPER")    return -600;
    return -350;
  };

  const dist = (s: SimNode, _t: SimNode) => {
    if (s.isSubject)           return 520;
    if (s.type === "TOPIC")    return 420;
    if (s.type === "SUBTOPIC") return 320;
    if (s.isArea)              return 240;
    if (s.isSubArea)           return 215;
    if (s.isCluster)           return 190;
    if (s.type === "PAPER")    return 160;
    return 130;
  };

  const sim = d3.forceSimulation<SimNode>(simNodes)
    // Stronger link force keeps children visually attached to their parent —
    // 0.4 was too weak and let many-sibling families drift apart.
    .force("link", d3.forceLink<SimNode, d3.SimulationLinkDatum<SimNode>>(links)
      .id(d => d.id).distance(l => dist(l.source as SimNode, l.target as SimNode)).strength(0.75))
    .force("charge", d3.forceManyBody<SimNode>().strength(d => repel(d)))
    .force("collide", d3.forceCollide<SimNode>().radius(d => d.r + 28).strength(0.9).iterations(6))
    .force("centerX", d3.forceX(0).strength(0.03))
    .force("tierY", d3.forceY<SimNode>(d => _nodeTierY(d)).strength(d =>
      d.type === "TOPIC" || d.type === "SUBTOPIC" ? 0.9 : 0.55
    ))
    .stop();

  for (let i = 0; i < 500; i++) sim.tick();

  const pos = new Map<string, {x:number;y:number}>();
  for (const n of simNodes) pos.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
  return pos;
}

// ─── Concept-node deduplication ────────────────────────────────────────────────
//
// The deep-build LLM can produce inconsistent label casing across batches/runs
// ("Task-Specific Assistants" vs "task-specific assistants"). The backend's
// get_or_create_node does an exact case-sensitive match, so each casing variant
// becomes a separate DB node with its own children. Visually this appears as
// "stray" duplicate cluster nodes floating disconnected from the canonical tree.
//
// This frontend dedup merges such duplicates: pick a canonical node per
// (lowercase label, namespace_key) group, redirect all edges from aliases to
// the canonical, and drop the alias nodes. It's purely cosmetic — the DB is
// untouched. A Clear All + Build Deep would also clean it, but cannot prevent
// LLM inconsistency within a single run, so this defensive dedup runs always.
function dedupeConceptNodes(
  nodes: GraphNode[],
  edges: GraphEdge[],
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  // Group CONCEPT nodes by (lowercase trimmed label | namespace_key)
  const groups = new Map<string, GraphNode[]>();
  for (const n of nodes) {
    if (n.type !== "CONCEPT") continue;
    const key = `${(n.label || "").trim().toLowerCase()}|${n.namespace_key || ""}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(n);
  }

  // Build alias→canonical map. Prefer the variant with at least one uppercase
  // letter (proper case). Fall back to the longest label, then alphabetical.
  const aliasToCanonical = new Map<string, string>();
  for (const [, group] of groups) {
    if (group.length <= 1) continue;
    const canonical = group.reduce((best, cur) => {
      const bestHasUpper = /[A-Z]/.test(best.label || "");
      const curHasUpper = /[A-Z]/.test(cur.label || "");
      if (curHasUpper && !bestHasUpper) return cur;
      if (!curHasUpper && bestHasUpper) return best;
      if ((cur.label || "").length !== (best.label || "").length) {
        return (cur.label || "").length > (best.label || "").length ? cur : best;
      }
      return (cur.label || "") < (best.label || "") ? cur : best;
    });
    for (const n of group) {
      if (n.id !== canonical.id) aliasToCanonical.set(n.id, canonical.id);
    }
  }

  if (aliasToCanonical.size === 0) return { nodes, edges };

  // Drop alias nodes
  const filteredNodes = nodes.filter(n => !aliasToCanonical.has(n.id));

  // Rewrite edges: redirect alias endpoints to canonical. Drop self-loops and
  // duplicates that result from rewriting.
  const seen = new Set<string>();
  const rewrittenEdges: GraphEdge[] = [];
  for (const e of edges) {
    const source = aliasToCanonical.get(e.source) || e.source;
    const target = aliasToCanonical.get(e.target) || e.target;
    if (source === target) continue;
    const key = `${source}->${target}->${e.type}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rewrittenEdges.push({ ...e, source, target });
  }

  return { nodes: filteredNodes, edges: rewrittenEdges };
}

// ─── Build ReactFlow graph ─────────────────────────────────────────────────────

function buildGraph(
  allNodes: GraphNode[], allEdges: GraphEdge[],
  expandedIds: Set<string>, bookmarkedIds: Set<string>, matchedIds: Set<string>,
  areaConceptIds: Set<string> = new Set(),
  clusterConceptIds: Set<string> = new Set(),  // = belowAreaConcepts (sub-areas + clusters)
  paperFolderMap: Map<string, Set<string>> | null = null,
  subjectTopicIds: Set<string> = new Set(),
  domainTopicIds: Set<string>  = new Set(),
  buildRunning: boolean = false,  // when true, skip paper-descendant pruning so partial areas show
  subAreaConceptIds: Set<string> = new Set(),  // middle tier for Y-layout separation
): { rfNodes: Node[]; rfEdges: Edge[] } {
  const nodeMap = new Map(allNodes.map(n => [n.id, n]));
  const parents: Record<string, string[]>  = {};
  const children: Record<string, string[]> = {};
  for (const e of allEdges) {
    (children[e.source] ??= []).push(e.target);
    (parents[e.target]  ??= []).push(e.source);
  }

  const hasTopics    = allNodes.some(n => n.type === "TOPIC");
  const hasSubjects  = subjectTopicIds.size > 0;
  const visible = new Set<string>();

  // All intermediate concept nodes (area and cluster) — they gate paper visibility
  const allIntermediateConcepts = new Set([...areaConceptIds, ...clusterConceptIds]);

  // Papers that have a cluster concept parent — must wait for cluster to expand
  const papersWithClusterParent = new Set<string>();
  for (const n of allNodes) {
    if (n.type === "PAPER" && (parents[n.id] || []).some(p => clusterConceptIds.has(p))) {
      papersWithClusterParent.add(n.id);
    }
  }

  // Level order: process parents before children so visibility gates work correctly.
  // 8 levels: SUBJECT(0) → DOMAIN_TOPIC(1) → SUBTOPIC(2) → AREA(3) →
  //           CLUSTER(4) → PAPER(5) → LEAF_CONCEPT(6) → LEAF_METHOD(6)
  const getLevel = (n: GraphNode): number => {
    if (n.type === "TOPIC")          return subjectTopicIds.has(n.id) ? 0 : 1;
    if (n.type === "SUBTOPIC")       return 2;
    if (areaConceptIds.has(n.id))    return 3;
    if (clusterConceptIds.has(n.id)) return 4;
    if (n.type === "PAPER")          return 5;
    return 6; // leaf CONCEPT / METHOD
  };
  const orderedNodes = [...allNodes].sort((a, b) => getLevel(a) - getLevel(b));

  for (const n of orderedNodes) {
    if (hasTopics) {
      if (n.type === "TOPIC") {
        // Subject nodes (root): always visible.
        // Domain nodes (e.g. "Artificial Intelligence"): only when parent subject is expanded.
        if (hasSubjects) {
          if (subjectTopicIds.has(n.id)) {
            visible.add(n.id);
          } else if (domainTopicIds.has(n.id)) {
            const ps = parents[n.id] || [];
            if (ps.some(p => expandedIds.has(p) && visible.has(p))) visible.add(n.id);
          } else {
            // Orphan TOPIC (shouldn't happen after rebuild-hierarchy, but safe fallback)
            visible.add(n.id);
          }
        } else {
          // No subjects yet (pre-rebuild) — show all TOPIC nodes
          visible.add(n.id);
        }
        continue;
      }

      const ps = parents[n.id] || [];
      if (papersWithClusterParent.has(n.id)) {
        const clusterPs = ps.filter(p => clusterConceptIds.has(p));
        if (clusterPs.some(p => expandedIds.has(p) && visible.has(p))) visible.add(n.id);
      } else if (allIntermediateConcepts.has(n.id)) {
        if (ps.some(p => expandedIds.has(p) && visible.has(p))) visible.add(n.id);
      } else {
        // Never show a non-TOPIC node unless it has a visible, expanded parent.
        // The old `ps.length === 0` shortcut made orphan nodes (e.g. concepts
        // whose paper parents were excluded by the namespace filter) appear as
        // stray floating nodes with no connection to any hierarchy.
        if (ps.some(p => expandedIds.has(p) && visible.has(p))) visible.add(n.id);
      }
    } else {
      // Flat graph fallback (no TOPIC nodes built yet)
      if (n.type === "PAPER") { visible.add(n.id); continue; }
      const ps = parents[n.id] || [];
      if (ps.length === 0 || ps.some(p => expandedIds.has(p))) visible.add(n.id);
    }
  }

  // Force-reveal matched nodes and walk their ancestors up to the root so the
  // user can see where a searched node lives in the hierarchy.
  // Only walk structural (non-related_to) edges to avoid pulling in semantic siblings.
  if (matchedIds.size > 0) {
    const hierarchyEdges = allEdges.filter(e => e.type !== "related_to");
    for (const id of matchedIds) {
      visible.add(id);
      // Walk upward to the root so the matched node is reachable in the visible tree
      let frontier = [id];
      while (frontier.length) {
        const next: string[] = [];
        for (const nid of frontier) {
          for (const e of hierarchyEdges) {
            if (e.target === nid && !visible.has(e.source)) {
              visible.add(e.source);
              next.push(e.source);
            }
          }
        }
        frontier = next;
      }
    }
  }

  // ── Definitive stray-node eliminator ────────────────────────────────────────
  // After ALL visibility logic (including search force-reveal), ensure no
  // non-TOPIC visible node is disconnected from the visible hierarchy.
  // Iterate until stable so cascading orphans are handled:
  //   grandparent removed → parent stranded → child stranded → …
  // TOPIC nodes are hierarchy roots and are always kept.
  {
    let changed = true;
    while (changed) {
      changed = false;
      for (const id of [...visible]) {
        const n = nodeMap.get(id);
        if (!n || n.type === "TOPIC") continue;
        const ps = parents[id] || [];
        if (ps.length > 0 && !ps.some(p => visible.has(p))) {
          visible.delete(id);
          changed = true;
        }
      }
    }
  }

  // ── Prune: remove intermediate concept nodes that are completely childless.
  //
  // Previous approaches all had fatal flaws:
  //   v1 — visible.has(cid) guard: circular dep (area pruned ↔ children invisible)
  //   v2 — recursive traversal without guard: O(N²), froze renderer
  //   v3 — upward BFS from papers: missed area concepts because ingestion creates
  //        SUBTOPIC→PAPER edges (not CLUSTER→PAPER), so papers never propagate
  //        through the CONCEPT parent chain to reach area nodes.
  //
  // Correct rule: only prune an intermediate concept node if it has ZERO children
  // in the graph (children map uses all edges, not the visible set). This removes
  // genuine orphan/noise concept nodes while keeping any node that has structure
  // below it — even if that structure has no papers yet (e.g. areas whose cluster
  // → paper edges weren't created due to UUID mismatch in the build output).
  if (!buildRunning) {
    for (const id of [...visible]) {
      const n = nodeMap.get(id);                        // O(1) — was allNodes.find() O(N)
      if (!n || n.type !== "CONCEPT") continue;
      const isIntermediate = areaConceptIds.has(id) || clusterConceptIds.has(id) ||
        (parents[id] || []).some(p => areaConceptIds.has(p) || clusterConceptIds.has(p));
      if (isIntermediate && (children[id] || []).length === 0) {
        visible.delete(id);
      }
    }
  }

  // ── DAG enforcement: remove back-edges AND skip-level bypass edges.
  // Hierarchical edges must flow strictly from higher levels to lower levels.
  // related_to edges (dotted) are excluded — they're semantic, not structural.
  const levelOf = (n: typeof allNodes[0]): number => {
    if (subjectTopicIds.has(n.id))        return 0;
    if (n.type === "TOPIC")               return 1;
    if (n.type === "SUBTOPIC")            return 2;
    if (areaConceptIds.has(n.id))         return 3;
    if (subAreaConceptIds.has(n.id))      return 4;
    if (clusterConceptIds.has(n.id))      return 5;
    if (n.type === "PAPER")               return 6;
    return 7; // leaf CONCEPT / METHOD
  };
  const nodeLevel = new Map(allNodes.map(n => [n.id, levelOf(n)]));
  const dagEdges = allEdges.filter(e => {
    if (e.type === "related_to") return true; // keep semantic edges as-is
    const sl = nodeLevel.get(e.source) ?? -1;
    const tl = nodeLevel.get(e.target) ?? -1;
    if (sl >= tl) return false; // back-edge or same-level: remove (DAG enforcement)

    // Remove stale skip-level bypass edges to PAPER nodes.
    // add_paper_node creates SUBTOPIC→PAPER during ingestion; build_deep_graph later
    // creates CLUSTER→PAPER. The old direct edges are redundant and render as stray
    // lines. Remove any SUBTOPIC/AREA/SUBAREA→PAPER edge when the paper already has
    // a proper cluster-level parent in this graph.
    if (tl === 6 /* PAPER */) {
      const srcLevel = sl;
      if (srcLevel < 5 /* below cluster tier */) {
        const paperParents = (parents[e.target] || []);
        // If any cluster-tier node is a parent of this paper, the bypass edge is stale
        if (paperParents.some(p => clusterConceptIds.has(p))) return false;
      }
    }

    return true;
  });

  const pos = runForce(allNodes, dagEdges, visible, areaConceptIds, clusterConceptIds, subjectTopicIds, subAreaConceptIds);

  // Child count: type-aware to match what's actually shown at each level
  const effectiveChildCount = (nodeId: string, type: string): number => {
    const ch = children[nodeId] || [];
    if (type === "TOPIC") {
      // Subject node → count domain TOPIC children
      if (subjectTopicIds.has(nodeId)) {
        return ch.filter(cid => domainTopicIds.has(cid)).length;
      }
      // Domain node → count SUBTOPIC children
      return ch.filter(cid => nodeMap.get(cid)?.type === "SUBTOPIC").length;
    }
    if (type === "SUBTOPIC") {
      return ch.filter(cid => {
        if (areaConceptIds.has(cid)) return true;
        const child = nodeMap.get(cid);
        return child?.type === "PAPER" && !papersWithClusterParent.has(cid);
      }).length;
    }
    if (areaConceptIds.has(nodeId)) {
      // area concept: child count is # of cluster concepts
      return ch.filter(cid => clusterConceptIds.has(cid)).length;
    }
    return ch.length;
  };

  const rfNodes: Node[] = allNodes.filter(n => visible.has(n.id)).map(n => {
    const p = pos.get(n.id) ?? { x: 0, y: 0 };
    const isBookmarked = n.type === "PAPER" && !!n.paper_id && bookmarkedIds.has(n.paper_id);
    const isSubject = subjectTopicIds.has(n.id);
    const isArea    = areaConceptIds.has(n.id);
    const isCluster = clusterConceptIds.has(n.id);
    const r = n.type === "PAPER" ? undefined
      : isSubject ? 72
      : isArea ? 54 : isCluster ? 44 : (RADIUS[n.type] ?? 38);
    return {
      id: n.id,
      type: n.type === "PAPER" ? "paperNode" : "clusterNode",
      position: { x: p.x - (n.type === "PAPER" ? 76 : (r ?? 38)), y: p.y - (n.type === "PAPER" ? 40 : (r ?? 38)) },
      data: {
        label: n.label, type: n.type, paper_id: n.paper_id, namespace_key: n.namespace_key,
        description: n.description,
        isBookmarked, matched: matchedIds.has(n.id),
        childCount: effectiveChildCount(n.id, n.type),
        expanded: expandedIds.has(n.id),
        isSubject,
        isCluster: isArea || isCluster,  // subject nodes are NOT clusters
        r,
      },
      style: n.type !== "PAPER" ? { width: (r ?? 38) * 2, height: (r ?? 38) * 2, background: "transparent", border: "none" } : undefined,
    };
  });

  // Build paper_id → node_id map for cross-folder detection
  const paperNodeMap = new Map<string, string>();
  for (const n of allNodes) if (n.paper_id) paperNodeMap.set(n.paper_id, n.id);

  const visSet = new Set(rfNodes.map(n => n.id));
  const rfEdges: Edge[] = dagEdges.filter(e => visSet.has(e.source) && visSet.has(e.target)).map(e => {
    const isRelated   = e.type === "related_to";
    const isMatched   = matchedIds.has(e.source) || matchedIds.has(e.target);

    // Detect cross-folder: source and target papers are in different folders
    let isCrossFolder = false;
    if (paperFolderMap && !isRelated) {
      const srcNode = allNodes.find(n => n.id === e.source);
      const tgtNode = allNodes.find(n => n.id === e.target);
      if (srcNode?.paper_id && tgtNode?.paper_id) {
        const sf = paperFolderMap.get(srcNode.paper_id);
        const tf = paperFolderMap.get(tgtNode.paper_id);
        if (sf && tf) {
          isCrossFolder = ![...sf].some(fid => tf.has(fid));
        }
      }
    }

    // related_to: semi-transparent violet dotted, no arrow — implies "semantically similar"
    if (isRelated) {
      return {
        id: e.id, source: e.source, target: e.target, type: "default", animated: false,
        style: {
          stroke: isMatched ? "rgba(251,191,36,0.55)" : "rgba(139,92,246,0.30)",
          strokeWidth: 1,
          strokeDasharray: "4 5",
        },
        // No arrowhead — similarity is undirected
        label: undefined,
      };
    }

    return {
      id: e.id, source: e.source, target: e.target, type: "default", animated: false,
      style: {
        stroke: isMatched
          ? "rgba(251,191,36,0.6)"
          : isCrossFolder ? "rgba(251,146,60,0.6)"
          : e.cross_namespace ? "rgba(245,158,11,0.45)" : "rgba(55,65,81,0.4)",
        strokeWidth: isCrossFolder ? 1.5 : e.cross_namespace ? 1.5 : 1,
        strokeDasharray: isCrossFolder ? "6 4" : e.cross_namespace ? "5 4" : undefined,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: isCrossFolder ? "rgba(251,146,60,0.6)"
          : e.cross_namespace ? "rgba(245,158,11,0.5)" : "rgba(55,65,81,0.5)",
        width: 9, height: 9,
      },
      label: isCrossFolder ? "cross-folder" : undefined,
      labelStyle: isCrossFolder ? { fontSize: 8, fill: "rgba(251,146,60,0.7)" } : undefined,
    };
  });

  return { rfNodes, rfEdges };
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export default function GraphPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(false);
  const [allGNodes, setAllGNodes] = useState<GraphNode[]>([]);
  const [allGEdges, setAllGEdges] = useState<GraphEdge[]>([]);
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());
  const [expandedIds,   setExpandedIds]   = useState<Set<string>>(new Set());
  const [selectedNode,  setSelectedNode]  = useState<GraphNode | null>(null);
  const [bookmarksOnly, setBookmarksOnly] = useState(false); // Full Feed by default — bookmarks subset is opt-in
  // Graph build state — narrow selectors so this page doesn't re-render
  // on every JobsPanel poll (e.g. while a podcast generates in the background).
  const graphBuildJobs = useJobsStore((s) => s.graphBuildJobs);
  const addGraphBuildJob = useJobsStore((s) => s.addGraphBuildJob);
  const dismissGraphBuildJob = useJobsStore((s) => s.dismissGraphBuildJob);
  const activeBuildJob = graphBuildJobs.find(g => g.status === "running");
  const buildingDeep = !!activeBuildJob;

  const mountedRef    = useRef(true);
  const submittingRef = useRef(false);  // prevents double-click race before store update
  const [searchQ, setSearchQ] = useState("");
  const [matchedIds, setMatchedIds] = useState<Set<string>>(new Set());
  const [folders, setFolders] = useState<BookmarkFolder[]>([]);
  const [selectedFolderIds, setSelectedFolderIds] = useState<Set<string>>(new Set());
  // paper_id → Set of folder IDs it belongs to
  const [paperFolderMap, setPaperFolderMap] = useState<Map<string, Set<string>>>(new Map());
  const [showFolderMenu, setShowFolderMenu] = useState(false);
  const [folderMenuPos, setFolderMenuPos] = useState({ top: 0, left: 0 });
  const folderBtnRef = useRef<HTMLButtonElement>(null);
  const folderMenuRef = useRef<HTMLDivElement>(null);
  const rfRef = useRef<ReactFlowInstance | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const { getPrimaryNamespaceKey, selectedTopics } = useNamespaceStore();
  const activeNs = getPrimaryNamespaceKey();

  // Per-topic filter: "all" shows every selected topic; any namespace key narrows to one.
  // Reset to "all" whenever the topic selection changes.
  const [topicFilter, setTopicFilter] = useState<string>("all");
  const prevTopicsRef = useRef(selectedTopics.join(","));
  useEffect(() => {
    const key = selectedTopics.join(",");
    if (key !== prevTopicsRef.current) { prevTopicsRef.current = key; setTopicFilter("all"); }
  }, [selectedTopics]);

  // The namespace keys actually sent to the API — either all topics or a single filtered one.
  const activeTopics = topicFilter === "all" ? selectedTopics : [topicFilter];
  const topicsKey = activeTopics.join(",");

  // Load folders
  useEffect(() => {
    api.get<BookmarkFolder[]>("/bookmarks/folders").then(setFolders).catch(() => {});
  }, []);

  // Bookmarks + folder membership
  // When bookmarks change in Bookmarks mode, auto-reload the graph after a short
  // delay (to let the backend's _index_paper_background finish adding the node).
  const prevBookmarkCount = useRef<number | null>(null);
  useEffect(() => {
    api.get<Bookmark[]>("/bookmarks").then(data => {
      const ids = new Set<string>();
      const map = new Map<string, Set<string>>();
      for (const b of (Array.isArray(data) ? data : [])) {
        if (b.paper?.id) {
          ids.add(b.paper.id);
          if (b.folder_ids?.length) {
            map.set(b.paper.id, new Set(b.folder_ids));
          }
        }
      }

      // Record count so the dedicated auto-reload effect can detect additions
      prevBookmarkCount.current = ids.size;
      prevBookmarkCount.current = ids.size;

      setBookmarkedIds(ids);
      setPaperFolderMap(map);
    }).catch(() => {});
  }, [activeNs]);

  // Clean up on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Reload graph whenever a build job transitions from running → done/failed.
  // Also handles the "build completed while away" case: on mount, if any job
  // completed within the last 30 minutes, reload to pick up the fresh graph.
  const prevBuildJobs     = useRef(graphBuildJobs);
  const initialLoadDone   = useRef(false);
  useEffect(() => {
    if (!initialLoadDone.current) {
      // First run after mount — check for recently-finished jobs (< 30 min)
      initialLoadDone.current = true;
      const thirtyMinsAgo = Date.now() - 30 * 60 * 1000;
      const recentlyDone = graphBuildJobs.some(
        g => g.status === "done" && g.completed_at && new Date(g.completed_at).getTime() > thirtyMinsAgo
      );
      if (recentlyDone) {
        // Trigger a fresh load after the initial loadGraph() has fired
        setTimeout(() => { if (mountedRef.current) loadGraph(); }, 500);
      }
      prevBuildJobs.current = graphBuildJobs;
      return;
    }

    const prev = prevBuildJobs.current;
    const justFinished = graphBuildJobs.filter(g =>
      (g.status === "done" || g.status === "failed") &&
      prev.find(p => p.job_id === g.job_id)?.status === "running"
    );
    if (justFinished.length > 0 && mountedRef.current) {
      loadGraph();
    }
    prevBuildJobs.current = graphBuildJobs;
  }, [graphBuildJobs]); // eslint-disable-line react-hooks/exhaustive-deps

  // Close folder menu on outside click
  useEffect(() => {
    if (!showFolderMenu) return;
    function handler(e: MouseEvent) {
      const target = e.target as HTMLElement | null;
      const insideBtn = folderBtnRef.current?.contains(target);
      const insideMenu = folderMenuRef.current?.contains(target);
      if (!insideBtn && !insideMenu) setShowFolderMenu(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showFolderMenu]);

  // Load graph
  const loadGraph = useCallback(async () => {
    setLoading(true);
    setExpandedIds(new Set());
    setSelectedNode(null);
    setSearchQ("");
    setMatchedIds(new Set());
    try {
      const bm = bookmarksOnly ? "&bookmarks_only=true" : "";
      const ns = activeTopics.length > 0
        ? `&namespace_keys=${encodeURIComponent(activeTopics.join(","))}`
        : "";
      const data = await api.get<GraphResponse>(`/graph?depth=3${bm}${ns}`);
      const deduped = dedupeConceptNodes(data.nodes, data.edges);
      setAllGNodes(deduped.nodes);
      setAllGEdges(deduped.edges);
    } catch {
      setAllGNodes([]); setAllGEdges([]);
    } finally { setLoading(false); }
  }, [topicsKey, bookmarksOnly]);

  useEffect(() => { loadGraph(); }, [loadGraph]);

  // While a build is running, reload graph data every 20 s to pick up incremental
  // area commits — but preserve the user's expanded nodes and selected node so
  // the view doesn't collapse on each refresh.
  const activeBuildJobRef = useRef(activeBuildJob);
  activeBuildJobRef.current = activeBuildJob;
  useEffect(() => {
    if (!activeBuildJob) return;
    const interval = setInterval(async () => {
      if (!mountedRef.current) return;
      // Soft reload: update nodes/edges without resetting UI state
      try {
        const bm = bookmarksOnly ? "&bookmarks_only=true" : "";
        const ns = activeTopics.length > 0
          ? `&namespace_keys=${encodeURIComponent(activeTopics.join(","))}`
          : "";
        const data = await api.get<GraphResponse>(`/graph?depth=3${bm}${ns}`);
        const deduped = dedupeConceptNodes(data.nodes, data.edges);
        setAllGNodes(deduped.nodes);
        setAllGEdges(deduped.edges);
      } catch { /* silent — don't disrupt the view on transient errors */ }
    }, 20_000);
    return () => clearInterval(interval);
  }, [!!activeBuildJob]); // eslint-disable-line react-hooks/exhaustive-deps

  // Bookmarks mode: auto-reload after a paper is bookmarked so the new graph
  // node (added by _index_paper_background) appears without a manual refresh.
  // Uses a 4s delay to allow the backend background task to complete.
  useEffect(() => {
    if (!bookmarksOnly) return;
    const count = bookmarkedIds.size;
    if (prevBookmarkCount.current !== null && count > prevBookmarkCount.current) {
      const timer = setTimeout(() => { if (mountedRef.current) loadGraph(); }, 4000);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookmarkedIds.size, bookmarksOnly]);

  const resetGraph = useCallback(async () => {
    setLoading(true);
    try {
      const ns = activeNs ? `?namespace_key=${activeNs}` : "";
      await api.post(`/graph/cleanup${ns}`);
      await loadGraph();
    } catch {
      setLoading(false);
    }
  }, [activeNs, loadGraph]);

  // Rebuild layout — apply folder filter when folder(s) are selected
  useEffect(() => {
    if (!allGNodes.length) { setNodes([]); setEdges([]); return; }

    let filteredNodes = allGNodes;
    let filteredEdges = allGEdges;

    if (selectedFolderIds.size > 0) {
      // Keep papers in at least one selected folder, then walk upward to ancestors only
      const folderPaperIds = new Set<string>();
      for (const [pid, fids] of paperFolderMap) {
        if ([...fids].some(fid => selectedFolderIds.has(fid))) {
          folderPaperIds.add(pid);
        }
      }
      const keepNodeIds = new Set<string>();
      for (const n of allGNodes) {
        if (n.type === "PAPER" && n.paper_id && folderPaperIds.has(n.paper_id)) {
          keepNodeIds.add(n.id);
        }
      }
      // Walk upward only (child → parent) to avoid pulling in sibling papers
      let changed = true;
      while (changed) {
        changed = false;
        for (const e of allGEdges) {
          if (keepNodeIds.has(e.target) && !keepNodeIds.has(e.source)) {
            keepNodeIds.add(e.source);
            changed = true;
          }
        }
      }
      filteredNodes = allGNodes.filter(n => keepNodeIds.has(n.id));
      filteredEdges = allGEdges.filter(e => keepNodeIds.has(e.source) && keepNodeIds.has(e.target));
    }

    // Detect 2 levels of intermediate concept nodes for the 6-level hierarchy.
    const subtopicIds = new Set(allGNodes.filter(n => n.type === "SUBTOPIC").map(n => n.id));

    // ── Detect node tiers ─────────────────────────────────────────────────────
    // TOPIC nodes now have two tiers:
    //   subjectTopicIds  = TOPIC nodes with no TOPIC parent  (e.g. "Computer Science")
    //   domainTopicIds   = TOPIC nodes with a TOPIC parent   (e.g. "Artificial Intelligence")
    // This lets us show only the subject root initially, and reveal domains on expand.
    const allTopicIds = new Set(filteredNodes.filter(n => n.type === "TOPIC").map(n => n.id));
    const subjectTopicIds = new Set<string>();
    const domainTopicIds  = new Set<string>();
    for (const n of filteredNodes) {
      if (n.type !== "TOPIC") continue;
      const hasTopicParent = filteredEdges.some(e => e.target === n.id && allTopicIds.has(e.source));
      if (hasTopicParent) domainTopicIds.add(n.id);
      else subjectTopicIds.add(n.id);
    }

    // Level 3 (AREA) — CONCEPT nodes that are direct children of SUBTOPIC
    const areaConceptIds = new Set<string>();
    for (const e of filteredEdges) {
      if (filteredNodes.find(n => n.id === e.source)?.type === "SUBTOPIC") {
        const tgt = filteredNodes.find(n => n.id === e.target);
        if (tgt?.type === "CONCEPT") areaConceptIds.add(e.target);
      }
    }
    // Level 4 (SUB-AREA) — CONCEPT nodes that are children of AREA concepts
    const subAreaConceptIds = new Set<string>();
    for (const e of filteredEdges) {
      if (areaConceptIds.has(e.source)) {
        const tgt = filteredNodes.find(n => n.id === e.target);
        if (tgt?.type === "CONCEPT") subAreaConceptIds.add(e.target);
      }
    }
    // Level 5 (CLUSTER) — CONCEPT nodes that are children of SUB-AREA concepts
    const clusterConceptIds = new Set<string>();
    for (const e of filteredEdges) {
      if (subAreaConceptIds.has(e.source)) {
        const tgt = filteredNodes.find(n => n.id === e.target);
        if (tgt?.type === "CONCEPT") clusterConceptIds.add(e.target);
      }
    }
    // Fallback: if no sub-areas exist, area children are the cluster tier
    if (subAreaConceptIds.size === 0) {
      for (const e of filteredEdges) {
        if (areaConceptIds.has(e.source)) {
          const tgt = filteredNodes.find(n => n.id === e.target);
          if (tgt?.type === "CONCEPT") clusterConceptIds.add(e.target);
        }
      }
    }

    // Pass ALL below-area intermediate concepts as the cluster tier so every
    // level (sub-area AND cluster) is treated as expandable inside buildGraph.
    // Previously only subAreaConceptIds was passed when 3 tiers existed, leaving
    // the actual level-5 cluster nodes unclassified → they were treated as leaf
    // concepts → clicks highlighted instead of expanding → papers never appeared.
    const belowAreaConcepts = new Set([...subAreaConceptIds, ...clusterConceptIds]);

    const { rfNodes, rfEdges } = buildGraph(
      filteredNodes, filteredEdges, expandedIds, bookmarkedIds, matchedIds,
      areaConceptIds, belowAreaConcepts,
      selectedFolderIds.size > 0 ? paperFolderMap : null,
      subjectTopicIds, domainTopicIds,
      !!activeBuildJob,
      subAreaConceptIds,  // middle tier for Y-layout separation
    );
    setNodes(rfNodes);
    setEdges(rfEdges);
  }, [allGNodes, allGEdges, expandedIds, bookmarkedIds, matchedIds, selectedFolderIds, paperFolderMap]);

  // Search
  const doSearch = useCallback((q: string) => {
    setSearchQ(q);
    if (!q.trim()) { setMatchedIds(new Set()); return; }
    const q_low = q.toLowerCase();
    const matches = allGNodes.filter(n => n.label.toLowerCase().includes(q_low));
    setMatchedIds(new Set(matches.map(n => n.id)));

    // Auto-expand parents
    setExpandedIds(prev => {
      const next = new Set(prev);
      for (const n of matches) {
        for (const e of allGEdges) { if (e.target === n.id) next.add(e.source); }
      }
      return next;
    });

    // Pan to first match
    setTimeout(() => {
      if (!rfRef.current || !matches.length) return;
      const nd = rfRef.current.getNode(matches[0].id);
      if (nd) {
        const cx = nd.position.x + 76;
        const cy = nd.position.y + 40;
        rfRef.current.setCenter(cx, cy, { zoom: 1.8, duration: 700 });
      }
    }, 150);
  }, [allGNodes, allGEdges]);

  // Node click
  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    const gn = allGNodes.find(n => n.id === node.id);
    if (!gn) return;
    setSelectedNode(gn);

    // Leaf CONCEPT/METHOD nodes don't expand but DO get highlighted so the
    // user gets visual feedback that the click registered.
    const isIntermediateConcept = !!(node.data?.isCluster);
    const isLeaf = (gn.type === "CONCEPT" || gn.type === "METHOD") && !isIntermediateConcept;

    if (isLeaf) {
      // Toggle gold-ring highlight on leaf CONCEPT/METHOD nodes
      setMatchedIds(prev => {
        const next = new Set(prev);
        if (next.has(gn.id)) next.delete(gn.id);
        else next.add(gn.id);
        return next;
      });
      return;
    }

    // PAPER nodes: toggle gold-ring highlight AND expand/collapse their concepts
    if (gn.type === "PAPER") {
      setMatchedIds(prev => {
        const next = new Set(prev);
        if (next.has(gn.id)) next.delete(gn.id);
        else next.add(gn.id);
        return next;
      });
    }

    // All expandable nodes: expand or collapse the subtree.
    // Only traverse structural hierarchy edges for collapse — skip related_to
    // (semantic dotted edges) so nodes in other branches are never accidentally
    // collapsed just because they're semantically linked to this subtree.
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(gn.id)) {
        const hierarchyEdges = allGEdges.filter(e => e.type !== "related_to");
        const collapse = new Set([gn.id]);
        let changed = true;
        while (changed) {
          changed = false;
          for (const e of hierarchyEdges) {
            if (collapse.has(e.source) && !collapse.has(e.target)) { collapse.add(e.target); changed = true; }
          }
        }
        collapse.forEach(id => next.delete(id));
      } else {
        next.add(gn.id);
      }
      return next;
    });
  }, [allGNodes, allGEdges]);

  // Stats
  const paperCount   = useMemo(() => allGNodes.filter(n => n.type === "PAPER").length, [allGNodes]);
  const conceptCount = useMemo(() => allGNodes.filter(n => n.type === "CONCEPT" || n.type === "METHOD").length, [allGNodes]);
  const visibleCount = nodes.length;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#060912" }}>

      {/* ── Toolbar ── */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "9px 14px", borderBottom: "1px solid rgba(255,255,255,0.05)",
        background: "rgba(6,9,18,0.97)", backdropFilter: "blur(12px)", flexShrink: 0,
      }}>
        {/* Title */}
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 12px rgba(99,102,241,0.4)",
          }}>
            <NetworkIcon size={13} color="white" />
          </div>
          <span style={{ fontSize: "13px", fontWeight: 700, color: "#e5e7eb" }}>Knowledge Graph</span>
        </div>

        {/* Search */}
        <div style={{
          flex: 1, maxWidth: 320, display: "flex", alignItems: "center", gap: 6,
          background: "rgba(17,24,39,0.8)", border: `1px solid ${searchQ ? "rgba(251,191,36,0.5)" : "rgba(55,65,81,0.5)"}`,
          borderRadius: 9, padding: "4px 10px", transition: "border-color 0.2s",
        }}>
          <SearchIcon size={12} color={searchQ ? "#fbbf24" : "#4b5563"} />
          <input
            ref={searchRef}
            value={searchQ}
            onChange={e => doSearch(e.target.value)}
            placeholder="Search nodes… (/ to focus)"
            style={{
              flex: 1, background: "none", border: "none", outline: "none",
              fontSize: "11px", color: searchQ ? "#fde68a" : "#9ca3af",
            }}
          />
          {searchQ && (
            <button onClick={() => doSearch("")} style={{ background: "none", border: "none", cursor: "pointer", color: "#6b7280", display: "flex" }}>
              <XIcon size={11} />
            </button>
          )}
          {matchedIds.size > 0 && (
            <span style={{ fontSize: "9px", color: "#fbbf24", fontWeight: 700, whiteSpace: "nowrap" }}>
              {matchedIds.size} match{matchedIds.size !== 1 ? "es" : ""}
            </span>
          )}
        </div>

        {/* Topic filter — only shown when 2+ topics are selected */}
        {selectedTopics.length > 1 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <select
              value={topicFilter}
              onChange={e => setTopicFilter(e.target.value)}
              title="Filter graph to a single topic"
              style={{
                padding: "3px 8px", borderRadius: 8, fontSize: "11px", fontWeight: 600,
                border: topicFilter !== "all" ? "1px solid rgba(99,102,241,0.5)" : "1px solid rgba(55,65,81,0.5)",
                background: topicFilter !== "all" ? "rgba(20,20,50,0.8)" : "rgba(17,24,39,0.6)",
                color: topicFilter !== "all" ? "#818cf8" : "#6b7280",
                cursor: "pointer", outline: "none",
              }}
            >
              <option value="all">All Topics</option>
              {selectedTopics.map(ns => (
                <option key={ns} value={ns}>{ns}</option>
              ))}
            </select>
          </div>
        )}

        {/* Bookmarks toggle */}
        <button
          onClick={() => setBookmarksOnly(b => !b)}
          title={bookmarksOnly ? "Showing bookmarked papers only — click to show full feed" : "Showing all feed papers — click to show bookmarks only"}
          style={{
            display: "flex", alignItems: "center", gap: 5, padding: "4px 10px",
            borderRadius: 8, border: `1px solid ${bookmarksOnly ? "rgba(34,197,94,0.35)" : "rgba(55,65,81,0.5)"}`,
            background: bookmarksOnly ? "rgba(6,22,10,0.8)" : "rgba(17,24,39,0.6)",
            color: bookmarksOnly ? "#4ade80" : "#6b7280", fontSize: "11px", fontWeight: 600, cursor: "pointer",
          }}
        >
          <BookmarkIcon size={10} fill={bookmarksOnly ? "#4ade80" : "none"} />
          {bookmarksOnly ? "Bookmarks" : "Full Feed"}
        </button>

        {/* Folder filter */}
        {folders.length > 0 && (
          <div style={{ position: "relative" }}>
            <button
              ref={folderBtnRef}
              onClick={() => {
                if (!showFolderMenu && folderBtnRef.current) {
                  const r = folderBtnRef.current.getBoundingClientRect();
                  setFolderMenuPos({ top: r.bottom + 6, left: r.left });
                }
                setShowFolderMenu(v => !v);
              }}
              style={{
                display: "flex", alignItems: "center", gap: 5, padding: "4px 10px",
                borderRadius: 8, cursor: "pointer", fontSize: "11px", fontWeight: 600,
                border: selectedFolderIds.size > 0 ? "1px solid rgba(251,146,60,0.4)" : "1px solid rgba(55,65,81,0.5)",
                background: selectedFolderIds.size > 0 ? "rgba(40,20,5,0.8)" : "rgba(17,24,39,0.6)",
                color: selectedFolderIds.size > 0 ? "#fb923c" : "#6b7280",
              }}
            >
              <FolderIcon size={10} />
              {selectedFolderIds.size > 0 ? `${selectedFolderIds.size} folder${selectedFolderIds.size > 1 ? "s" : ""}` : "Folders"}
            </button>

            {showFolderMenu && typeof document !== "undefined" && createPortal(
              <div ref={folderMenuRef} style={{
                position: "fixed", top: folderMenuPos.top, left: folderMenuPos.left, zIndex: 99999,
                background: "rgba(10,14,20,0.98)", border: "1px solid rgba(55,65,81,0.5)",
                borderRadius: 10, boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
                minWidth: 180, overflow: "hidden",
              }}>
                <div style={{ padding: "6px 10px", borderBottom: "1px solid rgba(55,65,81,0.2)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: "9px", fontWeight: 700, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.06em" }}>Filter by folder</span>
                  {selectedFolderIds.size > 0 && (
                    <button onClick={() => setSelectedFolderIds(new Set())} style={{ fontSize: "9px", color: "#fb923c", background: "none", border: "none", cursor: "pointer" }}>Clear</button>
                  )}
                </div>
                {folders.map(f => {
                  const checked = selectedFolderIds.has(f.id);
                  return (
                    <button key={f.id}
                      onClick={() => setSelectedFolderIds(prev => {
                        const next = new Set(prev);
                        next.has(f.id) ? next.delete(f.id) : next.add(f.id);
                        return next;
                      })}
                      style={{
                        width: "100%", display: "flex", alignItems: "center", gap: 8,
                        padding: "7px 10px", background: "none", border: "none", cursor: "pointer",
                        transition: "background 0.1s",
                      }}
                      onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.04)")}
                      onMouseLeave={e => (e.currentTarget.style.background = "none")}
                    >
                      <div style={{ width: 8, height: 8, borderRadius: 2, flexShrink: 0, background: f.color || "#6366f1" }} />
                      <span style={{ flex: 1, fontSize: "11px", color: "#d1d5db", textAlign: "left", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</span>
                      <div style={{
                        width: 14, height: 14, borderRadius: 4, flexShrink: 0,
                        border: checked ? "none" : "1px solid #374151",
                        background: checked ? "#fb923c" : "transparent",
                        display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        {checked && <CheckIcon size={9} color="white" strokeWidth={3} />}
                      </div>
                    </button>
                  );
                })}
              </div>,
              document.body
            )}
          </div>
        )}

        <button
          onClick={loadGraph} disabled={loading}
          style={{
            width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
            borderRadius: 7, border: "1px solid rgba(55,65,81,0.5)", background: "rgba(17,24,39,0.6)",
            color: "#6b7280", cursor: "pointer",
          }}
          title="Refresh graph"
        >
          <RefreshCwIcon size={12} className={loading ? "animate-spin" : ""} />
        </button>

        <button
          onClick={async () => {
            if (!confirm("Clear ALL graph data? This cannot be undone.\nAny running Build Deep jobs will also be stopped.")) return;
            setLoading(true);
            try {
              // Dismiss all running/pending graph build jobs from the store so
              // the button re-enables and the notification is cleaned up.
              graphBuildJobs.forEach(g => dismissGraphBuildJob(g.job_id));
              // Scope the clear to the currently selected namespaces so other
              // namespaces' caches and locks are untouched.
              const ns = selectedTopics.length > 0
                ? `?namespace_keys=${encodeURIComponent(selectedTopics.join(","))}`
                : "";
              await api.post(`/graph/clear${ns}`);
              setAllGNodes([]); setAllGEdges([]);
            } catch {} finally { setLoading(false); }
          }}
          disabled={loading}
          style={{
            display: "flex", alignItems: "center", gap: 5, padding: "4px 10px",
            borderRadius: 8, border: "1px solid rgba(239,68,68,0.5)",
            background: "rgba(17,24,39,0.6)", color: "#ef4444",
            fontSize: "11px", fontWeight: 600, cursor: "pointer",
          }}
          title="Delete all graph nodes and edges. Also stops any running Build Deep jobs."
        >
          <Trash2Icon size={10} />
          Clear All
        </button>

        <button
          onClick={async () => {
            if (buildingDeep || submittingRef.current) return;
            submittingRef.current = true;
            try {
              // Build only the currently visible topics (respects single-topic filter)
              const topics = activeTopics.length > 0 ? activeTopics : [activeNs].filter(Boolean);
              if (!topics.length) return;

              // Single group_id ties all namespace jobs from this one click together
              const groupId = crypto.randomUUID();
              const now = new Date().toISOString();
              let queued = 0;

              for (const ns of topics) {
                const alreadyRunning = graphBuildJobs.some(
                  g => g.status === "running" && g.namespace_key === ns
                );
                if (alreadyRunning) continue;

                try {
                  const res = await api.post<{ job_id: string; status: string; namespace_key: string | null }>(
                    `/graph/build-deep-bg?namespace_key=${encodeURIComponent(ns)}`
                  );
                  const alreadyTracked = graphBuildJobs.some(g => g.job_id === res.job_id);
                  if (!alreadyTracked) {
                    addGraphBuildJob({
                      job_id: res.job_id,
                      namespace_key: ns || null,
                      status: "running",
                      message: null,
                      created_at: now,
                      completed_at: null,
                      group_id: groupId,
                      namespace_count: topics.length,
                    });
                    queued++;
                  }
                } catch { /* individual namespace failure — continue with others */ }
              }
              if (queued === 0) console.info("Build Deep: no new namespaces to build");
            } catch (err) {
              console.error("Build Deep failed to queue:", err);
            } finally {
              submittingRef.current = false;
            }
          }}
          disabled={loading || buildingDeep}
          style={{
            display: "flex", alignItems: "center", gap: 5, padding: "4px 10px",
            borderRadius: 8, border: "1px solid rgba(139,92,246,0.4)",
            background: "rgba(17,24,39,0.6)", color: "#a78bfa",
            fontSize: "11px", fontWeight: 600, cursor: buildingDeep ? "default" : "pointer",
          }}
          title={buildingDeep ? "Build Deep is running in background — see notifications for progress" : "Use LLM to generate deep research area → sub-area → cluster hierarchy (runs in background)"}
        >
          {buildingDeep
            ? <Loader2Icon size={10} className="animate-spin" />
            : <ZapIcon size={10} />}
          {buildingDeep ? "Building…" : "Build Deep"}
        </button>

        {/* Stats */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginLeft: "auto" }}>
          {[
            { dot: "#6366f1", label: `${paperCount} papers` },
            { dot: "#0d9488", label: `${conceptCount} concepts` },
            { dot: "#9ca3af", label: `${visibleCount} visible` },
          ].map(({ dot, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: dot }} />
              <span style={{ fontSize: "10px", color: "#374151", fontWeight: 500 }}>{label}</span>
            </div>
          ))}
          {expandedIds.size > 0 && (
            <button
              onClick={() => setExpandedIds(new Set())}
              style={{ fontSize: "9px", color: "#6366f1", background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 5, padding: "2px 7px", cursor: "pointer" }}
            >
              <EyeOffIcon size={9} style={{ display: "inline", marginRight: 3 }} />
              Collapse all
            </button>
          )}
        </div>
      </div>

      {/* ── Hint ── */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "4px 14px", borderBottom: "1px solid rgba(255,255,255,0.03)",
        background: "rgba(6,9,18,0.7)", flexShrink: 0,
      }}>
        <span style={{ fontSize: "9px", color: "#1f2937" }}>
          Subject → Domain → Subtopic → Area → Sub-area → Cluster → Papers → Concepts & Methods · click to drill deeper · violet dotted = semantically related · / to search
        </span>
        {searchQ && matchedIds.size === 0 && (
          <span style={{ fontSize: "9px", color: "#ef4444" }}>No nodes matched "{searchQ}"</span>
        )}
      </div>

      {/* ── Canvas ── */}
      <div style={{ flex: 1, position: "relative" }}>
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(6,9,18,0.88)", zIndex: 10 }}>
            <Loader2Icon className="animate-spin" size={26} color="#6366f1" style={{ marginBottom: 10 }} />
            <p style={{ fontSize: "11px", color: "#4b5563" }}>Computing force layout…</p>
          </div>
        )}

        {!loading && allGNodes.length === 0 ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 12 }}>
            <div style={{
              width: 64, height: 64, borderRadius: 16,
              background: "rgba(17,24,39,0.8)", border: "1px solid rgba(255,255,255,0.04)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 30px rgba(99,102,241,0.1)",
            }}>
              <NetworkIcon size={24} color="#374151" />
            </div>
            <p style={{ fontSize: "14px", fontWeight: 600, color: "#4b5563" }}>No graph data</p>
            <p style={{ fontSize: "12px", color: "#374151", maxWidth: 300, textAlign: "center", lineHeight: 1.5 }}>
              {bookmarksOnly
                ? "Switch to Full Feed mode to see the complete graph, or bookmark papers to populate Bookmarks view."
                : "Click Build Deep to generate the taxonomy, or refresh papers on the Feed first."}
            </p>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            onInit={inst => { rfRef.current = inst; }}
            fitView fitViewOptions={{ padding: 0.3 }}
            style={{ background: "#060912" }}
            minZoom={0.05} maxZoom={3}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} color="rgba(255,255,255,0.025)" gap={32} size={1} />
            <Controls style={{ background: "rgba(10,14,20,0.9)", border: "1px solid rgba(55,65,81,0.4)", borderRadius: 10 }} showInteractive={false} />

            {/* Legend */}
            <div style={{
              position: "absolute", bottom: 20, right: 14,
              background: "rgba(8,11,18,0.95)", border: "1px solid rgba(55,65,81,0.3)",
              borderRadius: 12, padding: "10px 12px",
              backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", zIndex: 5,
            }}>
              <p style={{ fontSize: "7.5px", fontWeight: 700, color: "#374151", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 7 }}>Legend</p>
              {(["PAPER", "CONCEPT", "METHOD", "TOPIC", "SUBTOPIC"] as const).map(t => (
                <div key={t} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
                  <div style={{ width: t === "PAPER" ? 10 : 8, height: t === "PAPER" ? 7 : 8, borderRadius: t === "PAPER" ? 2 : "50%", background: P[t].ring, boxShadow: `0 0 5px ${P[t].glow}` }} />
                  <span style={{ fontSize: "8.5px", color: "#4b5563" }}>{t.charAt(0) + t.slice(1).toLowerCase()}</span>
                </div>
              ))}
              <div style={{ borderTop: "1px solid rgba(55,65,81,0.2)", marginTop: 6, paddingTop: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
                  <div style={{ width: 10, height: 7, borderRadius: 2, background: "#fbbf24", boxShadow: "0 0 5px rgba(251,191,36,0.4)" }} />
                  <span style={{ fontSize: "8.5px", color: "#4b5563" }}>Search match</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 4 }}>
                  <div style={{ width: 14, height: 1, background: "none", border: "1px dashed rgba(139,92,246,0.6)" }} />
                  <span style={{ fontSize: "8.5px", color: "#4b5563" }}>Semantically related</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <div style={{ width: 14, height: 1, background: "none", border: "1px dashed rgba(251,146,60,0.7)" }} />
                  <span style={{ fontSize: "8.5px", color: "#4b5563" }}>Cross-folder link</span>
                </div>
              </div>
            </div>
          </ReactFlow>
        )}

        {/* Keyboard shortcut listener */}
        <KeyboardHandler onSearch={() => searchRef.current?.focus()} />

        {/* Node detail */}
        {selectedNode && (
          <NodePanel
            node={selectedNode}
            isExpanded={expandedIds.has(selectedNode.id)}
            isBookmarked={selectedNode.type === "PAPER" && !!selectedNode.paper_id && bookmarkedIds.has(selectedNode.paper_id)}
            isCluster={nodes.find(n => n.id === selectedNode.id)?.data?.isCluster ?? false}
            isSubject={nodes.find(n => n.id === selectedNode.id)?.data?.isSubject ?? false}
            relatedCount={allGEdges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).length}
            onClose={() => setSelectedNode(null)}
            onSearch={(q) => { doSearch(q); searchRef.current?.focus(); }}
            childCount={nodes.find(n => n.id === selectedNode.id)?.data?.childCount ?? 0}
            buildRunning={!!activeBuildJob}
          />
        )}
      </div>
    </div>
  );
}

// ─── Keyboard shortcut helper ──────────────────────────────────────────────────

function KeyboardHandler({ onSearch }: { onSearch: () => void }) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "/" && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault();
        onSearch();
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onSearch]);
  return null;
}

// ─── Node detail panel ─────────────────────────────────────────────────────────

function NodePanel({
  node, isExpanded, isBookmarked, isCluster, isSubject: isSubjectProp, relatedCount, childCount, buildRunning, onClose, onSearch,
}: {
  node: GraphNode; isExpanded: boolean; isBookmarked: boolean; isCluster: boolean;
  isSubject?: boolean;
  relatedCount: number; childCount: number; buildRunning?: boolean;
  onClose: () => void; onSearch: (q: string) => void;
}) {
  const type = node.type as keyof typeof P;
  const p = P[type] ?? P.PAPER;

  const isSubjectNode = !!isSubjectProp;

  const isTopLevel = isSubjectNode || node.type === "TOPIC" || node.type === "SUBTOPIC";
  const buildingHint = buildRunning && childCount === 0 && isTopLevel ? "Building taxonomy…" : null;

  const expandHint = buildingHint ?? (
    isSubjectNode            ? `${childCount} research area${childCount !== 1 ? "s" : ""}` :
    node.type === "TOPIC"    ? `${childCount} subtopic${childCount !== 1 ? "s" : ""}` :
    node.type === "SUBTOPIC" ? `${childCount} research area${childCount !== 1 ? "s" : ""}` :
    isCluster                ? `${childCount} ${childCount !== 1 ? "items" : "item"}` :
    node.type === "PAPER"    ? `${childCount} concept${childCount !== 1 ? "s" : ""} & method${childCount !== 1 ? "s" : ""}` :
    null
  );

  const isExpandable = isCluster || (node.type !== "CONCEPT" && node.type !== "METHOD");

  return (
    <div style={{
      position: "absolute", top: 14, right: 14, width: 288,
      background: "rgba(8,11,18,0.98)", border: "1px solid rgba(55,65,81,0.4)",
      borderRadius: 14, boxShadow: "0 8px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.04)",
      backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)",
      overflow: "hidden", zIndex: 20,
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 13px", borderBottom: "1px solid rgba(255,255,255,0.05)",
        background: `linear-gradient(90deg, ${p.badge}50, transparent)`,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: p.ring, boxShadow: `0 0 8px ${p.glow}` }} />
          <span style={{ fontSize: "9px", fontWeight: 800, color: p.text, textTransform: "uppercase", letterSpacing: "0.08em" }}>{node.type}</span>
          {isBookmarked && <BookmarkIcon size={10} color="#22c55e" fill="#22c55e" />}
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "#374151", display: "flex" }}>
          <XIcon size={13} />
        </button>
      </div>

      {/* Body */}
      <div style={{ padding: "13px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
        <p style={{ fontSize: "12px", fontWeight: 700, color: "#e5e7eb", lineHeight: 1.4 }}>{node.label}</p>

        {/* Description — 2-3 lines of context */}
        {node.description && (
          <p style={{
            fontSize: "10.5px", color: "#9ca3af", lineHeight: 1.6,
            borderLeft: `2px solid ${p.ring}40`, paddingLeft: 8,
          }}>
            {node.description}
          </p>
        )}

        {/* Expand / collapse hint */}
        {isExpandable && expandHint && (
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 4, height: 4, borderRadius: "50%", background: buildingHint ? "#f59e0b" : p.ring, opacity: 0.6 }} />
            <p style={{ fontSize: "9.5px", color: buildingHint ? "#d97706" : "#4b5563" }}>
              {buildingHint
                ? "Building taxonomy…"
                : isExpanded ? `Showing ${expandHint}` : `Click to explore ${expandHint}`}
            </p>
          </div>
        )}

        {/* Actions */}
        <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 2 }}>
          <button
            onClick={() => onSearch(node.label)}
            style={{
              display: "flex", alignItems: "center", gap: 6, padding: "6px 10px",
              borderRadius: 8, background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.2)",
              color: "#818cf8", fontSize: "10px", fontWeight: 600, cursor: "pointer",
            }}
          >
            <ZapIcon size={10} />
            Find related nodes
          </button>

          {node.type === "PAPER" && node.paper_id && (
            <button
              onClick={() => window.open(`/study/${node.paper_id}`, "_blank")}
              style={{
                display: "flex", alignItems: "center", gap: 6, padding: "6px 10px",
                borderRadius: 8, background: "rgba(13,148,136,0.1)", border: "1px solid rgba(13,148,136,0.2)",
                color: "#5eead4", fontSize: "10px", fontWeight: 600, cursor: "pointer",
              }}
            >
              <ExternalLinkIcon size={10} />
              Open in Study Mode
            </button>
          )}
          {node.type === "PAPER" && node.source_url && (
            <button
              onClick={() => window.open(node.source_url!, "_blank")}
              style={{
                display: "flex", alignItems: "center", gap: 6, padding: "6px 10px",
                borderRadius: 8, background: "rgba(251,146,60,0.08)", border: "1px solid rgba(251,146,60,0.2)",
                color: "#fb923c", fontSize: "10px", fontWeight: 600, cursor: "pointer",
              }}
            >
              <ExternalLinkIcon size={10} />
              arXiv
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
