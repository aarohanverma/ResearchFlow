import { create } from "zustand";
import { persist } from "zustand/middleware";

// ─── Full arXiv taxonomy ──────────────────────────────────────────────────────

export interface NsTopic   { key: string; label: string }
export interface NsSubject { key: string; label: string; color: string; icon: string; topics: NsTopic[] }

export const NAMESPACE_TREE: NsSubject[] = [
  {
    key: "cs", label: "Computer Science", color: "#6366f1", icon: "💻",
    topics: [
      { key: "cs.AI",  label: "Artificial Intelligence" },
      { key: "cs.AR",  label: "Hardware Architecture" },
      { key: "cs.CC",  label: "Computational Complexity" },
      { key: "cs.CE",  label: "Computational Engineering" },
      { key: "cs.CG",  label: "Computational Geometry" },
      { key: "cs.CL",  label: "Computation & Language" },
      { key: "cs.CR",  label: "Cryptography & Security" },
      { key: "cs.CV",  label: "Computer Vision" },
      { key: "cs.CY",  label: "Computers & Society" },
      { key: "cs.DB",  label: "Databases" },
      { key: "cs.DC",  label: "Distributed Computing" },
      { key: "cs.DL",  label: "Digital Libraries" },
      { key: "cs.DM",  label: "Discrete Mathematics" },
      { key: "cs.DS",  label: "Data Structures & Algorithms" },
      { key: "cs.ET",  label: "Emerging Technologies" },
      { key: "cs.FL",  label: "Formal Languages" },
      { key: "cs.GR",  label: "Graphics" },
      { key: "cs.GT",  label: "Game Theory" },
      { key: "cs.HC",  label: "Human-Computer Interaction" },
      { key: "cs.IR",  label: "Information Retrieval" },
      { key: "cs.IT",  label: "Information Theory" },
      { key: "cs.LG",  label: "Machine Learning" },
      { key: "cs.LO",  label: "Logic in CS" },
      { key: "cs.MA",  label: "Multiagent Systems" },
      { key: "cs.MS",  label: "Mathematical Software" },
      { key: "cs.NA",  label: "Numerical Analysis" },
      { key: "cs.NE",  label: "Neural & Evolutionary Computing" },
      { key: "cs.NI",  label: "Networking & Internet" },
      { key: "cs.OS",  label: "Operating Systems" },
      { key: "cs.PF",  label: "Performance" },
      { key: "cs.PL",  label: "Programming Languages" },
      { key: "cs.RO",  label: "Robotics" },
      { key: "cs.SE",  label: "Software Engineering" },
      { key: "cs.SI",  label: "Social & Information Networks" },
      { key: "cs.SY",  label: "Systems & Control" },
    ],
  },
  {
    key: "physics", label: "Physics", color: "#a78bfa", icon: "⚛️",
    topics: [
      { key: "quant-ph",          label: "Quantum Physics" },
      { key: "gr-qc",             label: "General Relativity & Quantum Cosmology" },
      { key: "hep-th",            label: "High Energy Physics – Theory" },
      { key: "hep-ph",            label: "High Energy Physics – Phenomenology" },
      { key: "hep-ex",            label: "High Energy Physics – Experiment" },
      { key: "math-ph",           label: "Mathematical Physics" },
      { key: "astro-ph.CO",       label: "Cosmology" },
      { key: "astro-ph.EP",       label: "Earth & Planetary Astrophysics" },
      { key: "astro-ph.GA",       label: "Galaxies" },
      { key: "astro-ph.HE",       label: "High Energy Astrophysics" },
      { key: "astro-ph.SR",       label: "Solar & Stellar Astrophysics" },
      { key: "cond-mat.stat-mech",label: "Statistical Mechanics" },
      { key: "cond-mat.str-el",   label: "Strongly Correlated Electrons" },
      { key: "cond-mat.supr-con", label: "Superconductivity" },
      { key: "cond-mat.mes-hall", label: "Mesoscale & Nanoscale Physics" },
      { key: "cond-mat.mtrl-sci", label: "Materials Science" },
      { key: "cond-mat.soft",     label: "Soft Condensed Matter" },
      { key: "cond-mat.dis-nn",   label: "Disordered Systems & Neural Networks" },
      { key: "physics.app-ph",    label: "Applied Physics" },
      { key: "physics.bio-ph",    label: "Biological Physics" },
      { key: "physics.chem-ph",   label: "Chemical Physics" },
      { key: "physics.comp-ph",   label: "Computational Physics" },
      { key: "physics.flu-dyn",   label: "Fluid Dynamics" },
      { key: "physics.optics",    label: "Optics" },
      { key: "physics.plasm-ph",  label: "Plasma Physics" },
      { key: "nlin.CD",           label: "Chaotic Dynamics" },
    ],
  },
  {
    key: "math", label: "Mathematics", color: "#10b981", icon: "∑",
    topics: [
      { key: "math.AC", label: "Commutative Algebra" },
      { key: "math.AG", label: "Algebraic Geometry" },
      { key: "math.AP", label: "Analysis of PDEs" },
      { key: "math.AT", label: "Algebraic Topology" },
      { key: "math.CA", label: "Classical Analysis & ODEs" },
      { key: "math.CO", label: "Combinatorics" },
      { key: "math.CT", label: "Category Theory" },
      { key: "math.DG", label: "Differential Geometry" },
      { key: "math.DS", label: "Dynamical Systems" },
      { key: "math.FA", label: "Functional Analysis" },
      { key: "math.GR", label: "Group Theory" },
      { key: "math.GT", label: "Geometric Topology" },
      { key: "math.LO", label: "Logic" },
      { key: "math.NA", label: "Numerical Analysis" },
      { key: "math.NT", label: "Number Theory" },
      { key: "math.OC", label: "Optimization & Control" },
      { key: "math.PR", label: "Probability" },
      { key: "math.QA", label: "Quantum Algebra" },
      { key: "math.RA", label: "Rings & Algebras" },
      { key: "math.RT", label: "Representation Theory" },
      { key: "math.ST", label: "Statistics Theory" },
    ],
  },
  {
    key: "stat", label: "Statistics", color: "#14b8a6", icon: "📊",
    topics: [
      { key: "stat.AP", label: "Applications" },
      { key: "stat.CO", label: "Computation" },
      { key: "stat.ME", label: "Methodology" },
      { key: "stat.ML", label: "Machine Learning" },
      { key: "stat.TH", label: "Theory" },
    ],
  },
  {
    key: "q-bio", label: "Quantitative Biology", color: "#84cc16", icon: "🧬",
    topics: [
      { key: "q-bio.BM", label: "Biomolecules" },
      { key: "q-bio.CB", label: "Cell Behavior" },
      { key: "q-bio.GN", label: "Genomics" },
      { key: "q-bio.MN", label: "Molecular Networks" },
      { key: "q-bio.NC", label: "Neurons & Cognition" },
      { key: "q-bio.PE", label: "Populations & Evolution" },
      { key: "q-bio.QM", label: "Quantitative Methods" },
    ],
  },
  {
    key: "eess", label: "Electrical Engineering", color: "#f59e0b", icon: "⚡",
    topics: [
      { key: "eess.AS", label: "Audio & Speech Processing" },
      { key: "eess.IV", label: "Image & Video Processing" },
      { key: "eess.SP", label: "Signal Processing" },
      { key: "eess.SY", label: "Systems & Control" },
    ],
  },
  {
    key: "econ", label: "Economics", color: "#fb923c", icon: "📈",
    topics: [
      { key: "econ.EM", label: "Econometrics" },
      { key: "econ.GN", label: "General Economics" },
      { key: "econ.TH", label: "Theoretical Economics" },
    ],
  },
  {
    key: "q-fin", label: "Quantitative Finance", color: "#22d3ee", icon: "💹",
    topics: [
      { key: "q-fin.CP", label: "Computational Finance" },
      { key: "q-fin.EC", label: "Economics" },
      { key: "q-fin.MF", label: "Mathematical Finance" },
      { key: "q-fin.PM", label: "Portfolio Management" },
      { key: "q-fin.RM", label: "Risk Management" },
      { key: "q-fin.ST", label: "Statistical Finance" },
      { key: "q-fin.TR", label: "Trading & Market Microstructure" },
    ],
  },
];

// ─── Lookup helpers ───────────────────────────────────────────────────────────

export const TOPIC_TO_SUBJECT: Record<string, NsSubject> = {};
for (const sub of NAMESPACE_TREE)
  for (const t of sub.topics) TOPIC_TO_SUBJECT[t.key] = sub;

export const ALL_TOPIC_KEYS = NAMESPACE_TREE.flatMap(s => s.topics.map(t => t.key));

export function subjectTopics(subjectKey: string): string[] {
  return NAMESPACE_TREE.find(s => s.key === subjectKey)?.topics.map(t => t.key) ?? [];
}

// ─── Store ────────────────────────────────────────────────────────────────────

interface NamespaceStore {
  /** Subjects the user has subscribed to — only these appear in the sidebar. */
  subscribedSubjects: string[];
  /** Currently active subject scope */
  activeSubject: string;
  /** Selected topic keys within the active subject (drives API calls) */
  selectedTopics: string[];
  /** Collapsed subjects in sidebar */
  collapsedSubjects: string[];
  /** Per-subject topic-selection memory — preserved when switching subjects so
   *  flipping back doesn't lose the user's curated topic set. */
  topicsBySubject: Record<string, string[]>;

  subscribeSubject:   (subjectKey: string) => void;
  unsubscribeSubject: (subjectKey: string) => void;
  setSubject:         (subjectKey: string) => void;
  toggleTopic:        (topicKey: string)   => void;
  selectAllTopics:    (subjectKey: string) => void;
  toggleSubjectCollapse: (subjectKey: string) => void;

  /** All topic keys to use in API calls (full subject if none explicitly selected). */
  getActiveNamespaceKeys: () => string[];
  /** First topic key — for APIs that take a single value. */
  getPrimaryNamespaceKey: () => string;
}

export const useNamespaceStore = create<NamespaceStore>()(
  persist(
    (set, get) => ({
      subscribedSubjects: ["cs"],
      activeSubject:  "cs",
      selectedTopics: ["cs.AI", "cs.LG"],
      collapsedSubjects: [],
      topicsBySubject: { cs: ["cs.AI", "cs.LG"] },

      subscribeSubject: (subjectKey) =>
        set(s => ({
          subscribedSubjects: s.subscribedSubjects.includes(subjectKey)
            ? s.subscribedSubjects
            : [...s.subscribedSubjects, subjectKey],
        })),

      unsubscribeSubject: (subjectKey) =>
        set(s => {
          const next = s.subscribedSubjects.filter(k => k !== subjectKey);
          // If we're removing the active subject, switch to the first remaining
          // and restore that subject's last selection (or fall back to the
          // first 2 topics for a fresh first-visit default).
          const newActive = s.activeSubject === subjectKey
            ? (next[0] ?? "cs")
            : s.activeSubject;
          const newTopics = s.activeSubject === subjectKey
            ? (s.topicsBySubject[newActive] ?? subjectTopics(newActive).slice(0, 2))
            : s.selectedTopics;
          // Drop the removed subject's memory so it gets defaulted on resubscribe.
          const { [subjectKey]: _gone, ...remainingMemory } = s.topicsBySubject;
          return {
            subscribedSubjects: next.length ? next : [subjectKey], // can't remove last
            activeSubject: newActive,
            selectedTopics: newTopics,
            topicsBySubject: remainingMemory,
          };
        }),

      setSubject: (subjectKey) => {
        const { topicsBySubject, activeSubject, selectedTopics } = get();
        // Persist current subject's selection before switching so we can
        // restore it next time the user comes back.
        const persistCurrent = activeSubject && selectedTopics.length > 0
          ? { ...topicsBySubject, [activeSubject]: selectedTopics }
          : topicsBySubject;
        // Restore the target subject's prior selection if we have one;
        // otherwise default to the first 3 topics for a sensible cold start.
        const restored = persistCurrent[subjectKey] ?? subjectTopics(subjectKey).slice(0, 3);
        set({
          activeSubject: subjectKey,
          selectedTopics: restored,
          topicsBySubject: persistCurrent,
        });
      },

      toggleTopic: (topicKey) => {
        const { selectedTopics, activeSubject, topicsBySubject } = get();
        const subject = TOPIC_TO_SUBJECT[topicKey];
        if (!subject || subject.key !== activeSubject) return;
        const has = selectedTopics.includes(topicKey);
        const candidate = has
          ? selectedTopics.filter(k => k !== topicKey)
          : [...selectedTopics, topicKey];
        const next = candidate.length ? candidate : [topicKey];
        set({
          selectedTopics: next,
          topicsBySubject: { ...topicsBySubject, [activeSubject]: next },
        });
      },

      selectAllTopics: (subjectKey) => {
        const all = subjectTopics(subjectKey);
        set(s => ({
          selectedTopics: all,
          topicsBySubject: { ...s.topicsBySubject, [subjectKey]: all },
        }));
      },

      toggleSubjectCollapse: (subjectKey) =>
        set(s => ({
          collapsedSubjects: s.collapsedSubjects.includes(subjectKey)
            ? s.collapsedSubjects.filter(k => k !== subjectKey)
            : [...s.collapsedSubjects, subjectKey],
        })),

      getActiveNamespaceKeys: () => get().selectedTopics,

      getPrimaryNamespaceKey: () => get().selectedTopics[0] ?? "cs.AI",
    }),
    {
      name: "rf-namespace-v4",
      // Migrate: prior versions persisted only `selectedTopics` flat. Seed
      // `topicsBySubject` from the snapshot so the first switch doesn't blank.
      migrate: (persisted, _version) => {
        const p = (persisted ?? {}) as Partial<NamespaceStore>;
        if (!p.topicsBySubject && p.activeSubject && p.selectedTopics) {
          return {
            ...p,
            topicsBySubject: { [p.activeSubject]: p.selectedTopics },
          } as NamespaceStore;
        }
        return p as NamespaceStore;
      },
      version: 4,
    },
  )
);

// Backwards-compat alias
export const NAMESPACES = NAMESPACE_TREE.flatMap(s =>
  s.topics.map(t => ({ key: t.key, label: t.label, color: s.color }))
);
