"""GraphService — high-level graph operations built on GraphRepository."""

import json
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.graph import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType
from app.models.paper import Paper
from app.repositories.graph import GraphRepository

log = logging.getLogger(__name__)

# Broad domain labels — multiple namespaces can share one TOPIC
# Subject root labels — one per broad academic field.
# Each namespace rolls up: SUBJECT → TOPIC → SUBTOPIC → ...
_NS_TO_SUBJECT: dict[str, str] = {
    # Computer Science
    "cs.AI":   "Computer Science", "cs.LG":   "Computer Science",
    "cs.NE":   "Computer Science", "cs.CL":   "Computer Science",
    "cs.CV":   "Computer Science", "cs.GR":   "Computer Science",
    "cs.RO":   "Computer Science", "cs.SY":   "Computer Science",
    "cs.IR":   "Computer Science", "cs.DB":   "Computer Science",
    "cs.SE":   "Computer Science", "cs.PL":   "Computer Science",
    "cs.HC":   "Computer Science", "cs.CR":   "Computer Science",
    "cs.DC":   "Computer Science", "cs.GT":   "Computer Science",
    "cs.AR":   "Computer Science", "cs.CE":   "Computer Science",
    "cs.ET":   "Computer Science", "cs.MA":   "Computer Science",
    "cs.LV":   "Computer Science", "cs.MS":   "Computer Science",
    "cs.NA":   "Computer Science", "cs.OH":   "Computer Science",
    "cs.OS":   "Computer Science", "cs.PF":   "Computer Science",
    "cs.CC":   "Computer Science", "cs.CG":   "Computer Science",
    "cs.CY":   "Computer Science", "cs.DL":   "Computer Science",
    "cs.DM":   "Computer Science", "cs.DS":   "Computer Science",
    "cs.FL":   "Computer Science", "cs.IT":   "Computer Science",
    "cs.LO":   "Computer Science", "cs.NI":   "Computer Science",
    "cs.SI":   "Computer Science",
    # Mathematics
    "math.AC": "Mathematics", "math.AG": "Mathematics", "math.AP": "Mathematics",
    "math.AT": "Mathematics", "math.CA": "Mathematics", "math.CO": "Mathematics",
    "math.CT": "Mathematics", "math.DG": "Mathematics", "math.DS": "Mathematics",
    "math.FA": "Mathematics", "math.GR": "Mathematics", "math.GT": "Mathematics",
    "math.LO": "Mathematics", "math.NA": "Mathematics", "math.NT": "Mathematics",
    "math.OC": "Mathematics", "math.PR": "Mathematics", "math.QA": "Mathematics",
    "math.RA": "Mathematics", "math.RT": "Mathematics", "math.ST": "Mathematics",
    "math-ph": "Mathematics",
    # Physics
    "quant-ph":           "Physics", "gr-qc":              "Physics",
    "hep-th":             "Physics", "hep-ph":             "Physics",
    "hep-ex":             "Physics", "hep-lat":            "Physics",
    "astro-ph.CO":        "Physics", "astro-ph.EP":        "Physics",
    "astro-ph.GA":        "Physics", "astro-ph.HE":        "Physics",
    "astro-ph.IM":        "Physics", "astro-ph.SR":        "Physics",
    "cond-mat.stat-mech": "Physics", "cond-mat.str-el":    "Physics",
    "cond-mat.supr-con":  "Physics", "cond-mat.mes-hall":  "Physics",
    "cond-mat.mtrl-sci":  "Physics", "cond-mat.soft":      "Physics",
    "cond-mat.dis-nn":    "Physics", "cond-mat.quant-gas": "Physics",
    "physics.app-ph":     "Physics", "physics.bio-ph":     "Physics",
    "physics.chem-ph":    "Physics", "physics.comp-ph":    "Physics",
    "physics.flu-dyn":    "Physics", "physics.optics":     "Physics",
    "physics.plasm-ph":   "Physics", "physics.atom-ph":    "Physics",
    "nlin.AO":            "Physics", "nlin.CD":            "Physics",
    "nlin.CG":            "Physics", "nlin.PS":            "Physics",
    "nlin.SI":            "Physics",
    # Statistics & Machine Learning
    "stat.ML": "Statistics & Machine Learning", "stat.ME": "Statistics & Machine Learning",
    "stat.TH": "Statistics & Machine Learning", "stat.AP": "Statistics & Machine Learning",
    "stat.CO": "Statistics & Machine Learning", "stat.OT": "Statistics & Machine Learning",
    # Electrical Engineering
    "eess.SP": "Electrical Engineering & Signal Processing",
    "eess.AS": "Electrical Engineering & Signal Processing",
    "eess.IV": "Electrical Engineering & Signal Processing",
    "eess.SY": "Electrical Engineering & Signal Processing",
    # Biology
    "q-bio":      "Computational Biology", "q-bio.BM": "Computational Biology",
    "q-bio.CB":   "Computational Biology", "q-bio.GN": "Computational Biology",
    "q-bio.MN":   "Computational Biology", "q-bio.NC": "Computational Biology",
    "q-bio.OT":   "Computational Biology", "q-bio.PE": "Computational Biology",
    "q-bio.QM":   "Computational Biology", "q-bio.SC": "Computational Biology",
    "q-bio.TO":   "Computational Biology",
    # Economics
    "econ.EM":    "Economics", "econ.GN": "Economics", "econ.TH": "Economics",
}

_NS_TO_TOPIC: dict[str, str] = {
    # Computer Science
    "cs.AI":   "Artificial Intelligence",    "cs.LG":   "Artificial Intelligence",
    "cs.NE":   "Artificial Intelligence",    "stat.ML": "Artificial Intelligence",
    "cs.CL":   "Natural Language Processing","cs.CV":   "Computer Vision",
    "cs.GR":   "Computer Vision",            "cs.RO":   "Robotics & Control",
    "cs.SY":   "Robotics & Control",         "eess.SP": "Signal Processing",
    "eess.AS": "Audio & Speech",             "eess.IV": "Computer Vision",
    "eess.SY": "Robotics & Control",         "cs.IR":   "Information Retrieval",
    "cs.DB":   "Databases & Data",           "cs.SE":   "Software Engineering",
    "cs.PL":   "Programming Languages",      "cs.HC":   "Human-Computer Interaction",
    "cs.CR":   "Security & Privacy",         "cs.DC":   "Distributed Systems",
    "cs.GT":   "Game Theory & Economics",    "cs.AR":   "Hardware Architecture",
    "cs.CE":   "Computational Engineering",  "cs.ET":   "Emerging Technologies",
    "cs.MA":   "Multiagent Systems",         "cs.LV":   "Logic in Computer Science",
    "cs.MS":   "Mathematical Software",      "cs.NA":   "Numerical Analysis",
    "cs.OH":   "Other Computer Science",     "cs.OS":   "Operating Systems",
    "cs.PF":   "Performance",                "cs.CC":   "Computational Complexity",
    "cs.CG":   "Computational Geometry",     "cs.CY":   "Computers & Society",
    "cs.DL":   "Digital Libraries",          "cs.DM":   "Discrete Mathematics",
    "cs.DS":   "Algorithms & Data Structures","cs.FL":  "Formal Languages",
    "cs.IT":   "Information Theory",         "cs.LO":   "Logic in Computer Science",
    "cs.NI":   "Networking & Internet",      "cs.SI":   "Social & Information Networks",
    # Mathematics
    "math.AC": "Algebra",                    "math.AG": "Algebraic Geometry",
    "math.AP": "Mathematical Analysis",      "math.AT": "Topology",
    "math.CA": "Mathematical Analysis",      "math.CO": "Combinatorics",
    "math.CT": "Algebra",                    "math.DG": "Geometry",
    "math.DS": "Dynamical Systems",          "math.FA": "Mathematical Analysis",
    "math.GR": "Algebra",                    "math.GT": "Topology",
    "math.LO": "Logic & Foundations",        "math.NA": "Numerical Methods",
    "math.NT": "Number Theory",              "math.OC": "Optimization",
    "math.PR": "Probability & Statistics",   "math.QA": "Algebra",
    "math.RA": "Algebra",                    "math.RT": "Algebra",
    "math.ST": "Probability & Statistics",   "math-ph": "Mathematical Physics",
    # Physics
    "quant-ph":           "Quantum Physics",         "gr-qc":             "Relativity & Cosmology",
    "hep-th":             "High Energy Physics",      "hep-ph":            "High Energy Physics",
    "hep-ex":             "High Energy Physics",      "hep-lat":           "High Energy Physics",
    "astro-ph.CO":        "Astrophysics",             "astro-ph.EP":       "Astrophysics",
    "astro-ph.GA":        "Astrophysics",             "astro-ph.HE":       "Astrophysics",
    "astro-ph.IM":        "Astrophysics",             "astro-ph.SR":       "Astrophysics",
    "cond-mat.stat-mech": "Condensed Matter",         "cond-mat.str-el":   "Condensed Matter",
    "cond-mat.supr-con":  "Condensed Matter",         "cond-mat.mes-hall": "Condensed Matter",
    "cond-mat.mtrl-sci":  "Condensed Matter",         "cond-mat.soft":     "Condensed Matter",
    "cond-mat.dis-nn":    "Condensed Matter",         "cond-mat.quant-gas":"Condensed Matter",
    "physics.app-ph":     "Applied Physics",          "physics.bio-ph":    "Biological Physics",
    "physics.chem-ph":    "Chemical Physics",         "physics.comp-ph":   "Computational Physics",
    "physics.flu-dyn":    "Fluid Dynamics",           "physics.optics":    "Optics & Photonics",
    "physics.plasm-ph":   "Plasma Physics",           "physics.atom-ph":   "Atomic Physics",
    "nlin.AO":            "Nonlinear Science",        "nlin.CD":           "Nonlinear Science",
    "nlin.CG":            "Nonlinear Science",        "nlin.PS":           "Nonlinear Science",
    "nlin.SI":            "Nonlinear Science",
    # Statistics
    "stat.ME": "Statistics",                "stat.TH": "Statistics",
    "stat.AP": "Statistics",                "stat.CO": "Statistics",
    "stat.OT": "Statistics",
    # Biology
    "q-bio":    "Computational Biology",    "q-bio.BM": "Bioinformatics",
    "q-bio.CB": "Cell Biology",             "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Biology",        "q-bio.NC": "Neuroscience",
    "q-bio.OT": "Computational Biology",    "q-bio.PE": "Evolutionary Biology",
    "q-bio.QM": "Computational Biology",    "q-bio.SC": "Systems Biology",
    "q-bio.TO": "Tissue & Organ Biology",
    # Economics
    "econ.EM":  "Econometrics",             "econ.GN": "Economics",
    "econ.TH":  "Economic Theory",
}

# Human-readable labels for namespace SUBTOPIC nodes
_TOPIC_DESC: dict[str, str] = {
    "Artificial Intelligence":       "Systems that perceive, reason, learn, and act — spanning machine learning, planning, and intelligent agents.",
    "Natural Language Processing":   "Computational methods for understanding, generating, and transforming human language at scale.",
    "Computer Vision":               "Algorithms that interpret and understand visual data from images, video, and 3D environments.",
    "Robotics & Control":            "Physical agents that perceive and act in the world — motion planning, manipulation, and autonomy.",
    "Signal Processing":             "Techniques for analyzing, transforming, and synthesizing signals across time and frequency domains.",
    "Audio & Speech":                "Models and systems for speech recognition, synthesis, separation, and audio understanding.",
    "Information Retrieval":         "Techniques for indexing, ranking, and retrieving relevant information from large corpora.",
    "Databases & Data":              "Principles and systems for storing, querying, and managing structured and unstructured data.",
    "Software Engineering":          "Methodologies, tools, and practices for designing, building, and maintaining software at scale.",
    "Programming Languages":         "Formal languages, compilers, type systems, and program analysis and verification.",
    "Human-Computer Interaction":    "Design and evaluation of interfaces, experiences, and interactions between humans and machines.",
    "Security & Privacy":            "Cryptographic protocols, threat modeling, and privacy-preserving computation.",
    "Distributed Systems":           "Coordination, consensus, and fault tolerance across networked machines.",
    "Game Theory & Economics":       "Strategic interaction, mechanism design, and algorithmic market modeling.",
    "Optimization":                  "Mathematical methods for finding optimal solutions under constraints — convex, combinatorial, and stochastic.",
    "Computational Biology":         "Computational methods for biological data, genomics, protein structure, and systems biology.",
    "Hardware Architecture":         "Processor design, memory hierarchies, accelerators, and computer organization.",
    "Computational Engineering":     "Numerical methods and software for scientific simulation and engineering applications.",
    "Emerging Technologies":         "Novel computing paradigms including quantum, neuromorphic, and bio-inspired systems.",
    "Multiagent Systems":            "Cooperative and competitive multi-agent environments, coordination, and collective intelligence.",
    "Logic in Computer Science":     "Formal verification, model checking, automated reasoning, and logical frameworks.",
    "Research":                      "Interdisciplinary research at the frontier of computing and science.",
}

_NS_DESC: dict[str, str] = {
    "cs.AI":   "Reasoning, planning, knowledge representation, search, and intelligent agent architectures.",
    "cs.LG":   "Statistical learning algorithms, generalization theory, neural network training, and benchmarks.",
    "cs.NE":   "Neural computation, evolutionary strategies, and brain-inspired learning architectures.",
    "stat.ML": "Probabilistic models, Bayesian inference, and statistical learning theory.",
    "cs.CL":   "Language models, parsing, machine translation, dialogue systems, and semantic understanding.",
    "cs.CV":   "Image recognition, object detection, segmentation, generation, and 3D scene understanding.",
    "cs.GR":   "Rendering, geometry processing, animation, and 3D reconstruction.",
    "cs.RO":   "Robot perception, motion planning, manipulation, and autonomous navigation.",
    "cs.SY":   "Control theory, dynamical systems, stability analysis, and feedback mechanisms.",
    "eess.SP": "Filter design, spectral analysis, signal reconstruction, and time-frequency methods.",
    "eess.AS": "Automatic speech recognition, text-to-speech synthesis, and audio generation.",
    "cs.IR":   "Search engines, recommendation systems, relevance ranking, and dense retrieval.",
    "cs.DB":   "Query optimization, transaction processing, data modeling, and scalable storage.",
    "cs.SE":   "Program verification, testing, software architecture, code generation, and DevOps.",
    "cs.PL":   "Type theory, compilers, program analysis, runtime systems, and language design.",
    "cs.HC":   "Usability, accessibility, user studies, and human factors in system design.",
    "cs.CR":   "Public-key cryptography, secure computation, differential privacy, and threat analysis.",
    "cs.DC":   "Distributed consensus, fault-tolerant protocols, and scalable coordination.",
    "cs.GT":   "Nash equilibria, auction theory, mechanism design, and computational economics.",
    "cs.AR":   "CPU/GPU microarchitecture, memory systems, FPGAs, and emerging hardware accelerators.",
    "cs.CE":   "Finite-element methods, computational fluid dynamics, and scientific computing software.",
    "cs.ET":   "Quantum computing, neuromorphic chips, DNA computing, and unconventional computing models.",
    "cs.MA":   "Autonomous agents, distributed problem-solving, game-theoretic coordination, and swarms.",
    "cs.LV":   "Temporal/modal logic, satisfiability, proof assistants, and formal specification.",
    "cs.MS":   "Libraries and frameworks for numerical linear algebra, optimization, and scientific computing.",
    "cs.NA":   "Numerical stability, finite-difference methods, iterative solvers, and approximation theory.",
    "cs.OH":   "History of computing, general surveys, and cross-cutting computer science research.",
    "cs.OS":   "Kernel design, scheduling, memory management, virtualisation, and real-time systems.",
    "cs.PF":   "Benchmarking, profiling, workload characterisation, and performance modelling.",
    "math.OC": "Convex optimization, dynamic programming, optimal control, and variational methods.",
    "math.AC": "Commutative rings, ideals, modules, and algebraic structures.",
    "math.AG": "Schemes, sheaves, cohomology, algebraic curves, and birational geometry.",
    "math.AP": "Partial differential equations, regularity, existence, and well-posedness.",
    "math.AT": "Homotopy theory, homology, cohomology, and topological invariants.",
    "math.CA": "Real and complex analysis, ODEs, harmonic analysis, and measure theory.",
    "math.CO": "Graph theory, combinatorial optimization, counting, and discrete structures.",
    "math.CT": "Category theory, functors, natural transformations, and higher categories.",
    "math.DG": "Riemannian geometry, differential forms, manifolds, and curvature.",
    "math.DS": "Dynamical systems, ergodic theory, chaos, and bifurcations.",
    "math.FA": "Banach spaces, operator theory, spectral theory, and functional analysis.",
    "math.GR": "Group theory, group actions, representation theory, and Lie groups.",
    "math.GT": "Low-dimensional topology, knot theory, 3-manifolds, and geometric topology.",
    "math.LO": "Mathematical logic, model theory, proof theory, and set theory.",
    "math.NA": "Numerical methods, algorithms, error analysis, and scientific computing.",
    "math.NT": "Number fields, modular forms, L-functions, and arithmetic geometry.",
    "math.PR": "Probability theory, stochastic processes, martingales, and random fields.",
    "math.QA": "Quantum groups, Hopf algebras, and quantum symmetry.",
    "math.RA": "Non-commutative rings, algebras, and algebraic structures.",
    "math.RT": "Representation theory, Lie algebras, and character theory.",
    "math.ST": "Statistical theory, estimation, testing, and asymptotic analysis.",
    "math-ph": "Mathematical methods in physics, spectral theory, and field theory.",
    "quant-ph":           "Quantum information, quantum computing, entanglement, and quantum protocols.",
    "gr-qc":              "General relativity, gravitational waves, black holes, and quantum cosmology.",
    "hep-th":             "String theory, supersymmetry, conformal field theory, and quantum gravity.",
    "hep-ph":             "Standard model phenomenology, collider physics, and beyond-the-SM models.",
    "hep-ex":             "Experimental high energy physics, detectors, and collider experiments.",
    "astro-ph.CO":        "Large-scale structure, CMB, dark matter, dark energy, and cosmological simulations.",
    "astro-ph.GA":        "Galaxy formation, stellar dynamics, and galactic structure.",
    "astro-ph.HE":        "Gamma-ray bursts, neutron stars, black holes, and cosmic rays.",
    "astro-ph.SR":        "Stellar evolution, solar physics, and variable stars.",
    "cond-mat.stat-mech": "Statistical mechanics, phase transitions, and thermodynamic systems.",
    "cond-mat.str-el":    "Strongly correlated electrons, Mott insulators, and heavy fermions.",
    "cond-mat.supr-con":  "Superconductivity, Cooper pairs, Josephson junctions, and topological superconductors.",
    "cond-mat.mtrl-sci":  "Electronic structure, materials design, and computational materials science.",
    "physics.comp-ph":    "Molecular dynamics, Monte Carlo methods, and computational modeling.",
    "physics.flu-dyn":    "Turbulence, fluid simulation, and hydrodynamic instabilities.",
    "physics.optics":     "Laser physics, photonics, nonlinear optics, and imaging systems.",
    "stat.ME": "Causal inference, experimental design, and statistical modeling.",
    "stat.TH": "Decision theory, minimax bounds, and theoretical statistics.",
    "stat.AP": "Applied statistics, data analysis, and domain-specific statistical methods.",
    "eess.IV": "Image segmentation, super-resolution, video processing, and visual quality.",
    "eess.SY": "Systems and control theory, feedback, and stability analysis.",
    "q-bio":   "Sequence analysis, protein structure prediction, network biology, and genomics.",
    "q-bio.GN": "Genome analysis, gene regulation, and genomic variation.",
    "q-bio.NC": "Neural circuits, computational neuroscience, and brain connectivity.",
    "q-bio.SC": "Gene regulatory networks, metabolic models, and systems-level analysis.",
}

_NS_LABEL: dict[str, str] = {
    "cs.AI":   "cs.AI — Reasoning & Agent Systems",
    "cs.LG":   "cs.LG — Learning Algorithms & Theory",
    "cs.NE":   "cs.NE — Neural & Evolutionary Computing",
    "stat.ML": "stat.ML — Statistical Machine Learning",
    "cs.CL":   "cs.CL — Language & Dialogue Systems",
    "cs.CV":   "cs.CV — Visual Recognition & Generation",
    "cs.GR":   "cs.GR — Graphics & 3D Geometry",
    "cs.RO":   "cs.RO — Robotics & Autonomous Systems",
    "cs.SY":   "cs.SY — Control Theory & Dynamical Systems",
    "eess.SP": "eess.SP — Signal Processing & Analysis",
    "eess.AS": "eess.AS — Audio, Speech & Music",
    "cs.IR":   "cs.IR — Search & Recommendation",
    "cs.DB":   "cs.DB — Databases & Data Management",
    "cs.SE":   "cs.SE — Software Engineering & Verification",
    "cs.PL":   "cs.PL — Programming Languages & Compilers",
    "cs.HC":   "cs.HC — Human-Computer Interaction",
    "cs.CR":   "cs.CR — Cryptography & Security",
    "cs.DC":   "cs.DC — Distributed & Parallel Systems",
    "cs.GT":   "cs.GT — Algorithmic Game Theory",
    "cs.AR":   "cs.AR — Hardware Architecture",
    "cs.CE":   "cs.CE — Computational Engineering",
    "cs.ET":   "cs.ET — Emerging Technologies",
    "cs.MA":   "cs.MA — Multiagent Systems",
    "cs.LV":   "cs.LV — Logic in Computer Science",
    "cs.MS":   "cs.MS — Mathematical Software",
    "cs.NA":   "cs.NA — Numerical Analysis",
    "cs.OH":   "cs.OH — Other Computer Science",
    "cs.OS":   "cs.OS — Operating Systems",
    "cs.PF":   "cs.PF — Performance",
    "math.OC": "math.OC — Optimization & Optimal Control",
    "math.AC": "math.AC — Commutative Algebra",
    "math.AG": "math.AG — Algebraic Geometry",
    "math.AP": "math.AP — Analysis of PDEs",
    "math.AT": "math.AT — Algebraic Topology",
    "math.CA": "math.CA — Classical Analysis & ODEs",
    "math.CO": "math.CO — Combinatorics",
    "math.CT": "math.CT — Category Theory",
    "math.DG": "math.DG — Differential Geometry",
    "math.DS": "math.DS — Dynamical Systems",
    "math.FA": "math.FA — Functional Analysis",
    "math.GR": "math.GR — Group Theory",
    "math.GT": "math.GT — Geometric Topology",
    "math.LO": "math.LO — Logic",
    "math.NA": "math.NA — Numerical Analysis",
    "math.NT": "math.NT — Number Theory",
    "math.PR": "math.PR — Probability",
    "math.QA": "math.QA — Quantum Algebra",
    "math.RA": "math.RA — Rings & Algebras",
    "math.RT": "math.RT — Representation Theory",
    "math.ST": "math.ST — Statistics Theory",
    "math-ph": "math-ph — Mathematical Physics",
    "quant-ph":           "quant-ph — Quantum Physics",
    "gr-qc":              "gr-qc — General Relativity & Quantum Cosmology",
    "hep-th":             "hep-th — High Energy Physics (Theory)",
    "hep-ph":             "hep-ph — High Energy Physics (Phenomenology)",
    "hep-ex":             "hep-ex — High Energy Physics (Experiment)",
    "hep-lat":            "hep-lat — High Energy Physics (Lattice)",
    "astro-ph.CO":        "astro-ph.CO — Cosmology",
    "astro-ph.EP":        "astro-ph.EP — Earth & Planetary Astrophysics",
    "astro-ph.GA":        "astro-ph.GA — Galaxies",
    "astro-ph.HE":        "astro-ph.HE — High Energy Astrophysics",
    "astro-ph.IM":        "astro-ph.IM — Instrumentation & Methods",
    "astro-ph.SR":        "astro-ph.SR — Solar & Stellar Astrophysics",
    "cond-mat.stat-mech": "cond-mat.stat-mech — Statistical Mechanics",
    "cond-mat.str-el":    "cond-mat.str-el — Strongly Correlated Electrons",
    "cond-mat.supr-con":  "cond-mat.supr-con — Superconductivity",
    "cond-mat.mes-hall":  "cond-mat.mes-hall — Mesoscale & Nanoscale Physics",
    "cond-mat.mtrl-sci":  "cond-mat.mtrl-sci — Materials Science",
    "cond-mat.soft":      "cond-mat.soft — Soft Condensed Matter",
    "cond-mat.dis-nn":    "cond-mat.dis-nn — Disordered Systems & Neural Networks",
    "cond-mat.quant-gas": "cond-mat.quant-gas — Quantum Gases",
    "physics.app-ph":     "physics.app-ph — Applied Physics",
    "physics.bio-ph":     "physics.bio-ph — Biological Physics",
    "physics.chem-ph":    "physics.chem-ph — Chemical Physics",
    "physics.comp-ph":    "physics.comp-ph — Computational Physics",
    "physics.flu-dyn":    "physics.flu-dyn — Fluid Dynamics",
    "physics.optics":     "physics.optics — Optics",
    "physics.plasm-ph":   "physics.plasm-ph — Plasma Physics",
    "physics.atom-ph":    "physics.atom-ph — Atomic Physics",
    "nlin.CD":            "nlin.CD — Chaotic Dynamics",
    "nlin.AO":            "nlin.AO — Adaptation & Self-Organizing Systems",
    "nlin.CG":            "nlin.CG — Cellular Automata & Lattice Gases",
    "nlin.PS":            "nlin.PS — Pattern Formation & Solitons",
    "nlin.SI":            "nlin.SI — Exactly Solvable & Integrable Systems",
    "stat.ME": "stat.ME — Methodology",
    "stat.TH": "stat.TH — Statistics Theory",
    "stat.AP": "stat.AP — Applications",
    "stat.CO": "stat.CO — Computation",
    "stat.OT": "stat.OT — Other Statistics",
    "eess.IV": "eess.IV — Image & Video Processing",
    "eess.SY": "eess.SY — Systems & Control",
    "q-bio":   "q-bio — Computational & Systems Biology",
    "q-bio.BM": "q-bio.BM — Biomolecules",
    "q-bio.CB": "q-bio.CB — Cell Behavior",
    "q-bio.GN": "q-bio.GN — Genomics",
    "q-bio.MN": "q-bio.MN — Molecular Networks",
    "q-bio.NC": "q-bio.NC — Neurons & Cognition",
    "q-bio.OT": "q-bio.OT — Other Quantitative Biology",
    "q-bio.PE": "q-bio.PE — Populations & Evolution",
    "q-bio.QM": "q-bio.QM — Quantitative Methods",
    "q-bio.SC": "q-bio.SC — Subcellular Processes",
    "q-bio.TO": "q-bio.TO — Tissues & Organs",
    "econ.EM":  "econ.EM — Econometrics",
    "econ.GN":  "econ.GN — General Economics",
    "econ.TH":  "econ.TH — Theoretical Economics",
}


def _derive_subject(ns: str | None) -> str:
    """Derive a human-readable subject label from any arXiv namespace prefix."""
    if not ns:
        return "Research"
    prefix = ns.split(".")[0].split("-")[0]
    return {
        "cs":      "Computer Science",
        "math":    "Mathematics",
        "physics": "Physics",
        "astro":   "Physics",
        "cond":    "Physics",
        "quant":   "Physics",
        "hep":     "Physics",
        "gr":      "Physics",
        "nlin":    "Physics",
        "stat":    "Statistics & Machine Learning",
        "eess":    "Electrical Engineering & Signal Processing",
        "q":       "Computational Biology",
        "econ":    "Economics",
    }.get(prefix, prefix.capitalize() + " Research")


def _derive_topic(ns: str | None) -> str:
    """Derive a domain topic label from any arXiv namespace."""
    if not ns:
        return "Research"
    if ns in _NS_TO_TOPIC:
        return _NS_TO_TOPIC[ns]
    prefix = ns.split(".")[0]
    suffix = ns.split(".", 1)[1] if "." in ns else ns
    subject = _derive_subject(ns)
    return f"{subject} — {suffix.upper()}"


def _derive_subtopic_label(ns: str | None) -> str:
    """Derive a subtopic label from any arXiv namespace."""
    if not ns:
        return "General"
    if ns in _NS_LABEL:
        return _NS_LABEL[ns]
    return ns  # raw namespace key as fallback (e.g. "math.AG")


class GraphService:
    """High-level graph operations for building and querying the knowledge graph.

    Wraps ``GraphRepository`` with domain logic for maintaining the
    TOPIC → SUBTOPIC → CLUSTER → PAPER → CONCEPT/METHOD hierarchy.
    Exposes methods for adding papers, rebuilding clusters, and serving
    serializable subgraphs to the frontend.

    Class Attributes:
        _build_cache: Maps namespace key to the paper count observed at the
            last successful deep build, used to skip redundant LLM calls.
    """

    # namespace_key → paper count at last deep build; skips redundant LLM calls
    _build_cache: dict[str, int] = {}

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the service with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` shared by the repository and
                any direct queries made in this service.
        """
        self._repo = GraphRepository(db)
        self._db = db

    def _topic_label(self, namespace_key: str | None) -> str:
        """Return the human-readable topic label for a namespace key."""
        if namespace_key:
            return _NS_TO_TOPIC.get(namespace_key, namespace_key.split(".")[0].upper() + " Research")
        return "Research"

    def _subtopic_label(self, namespace_key: str | None) -> str:
        """Return the human-readable subtopic label for a namespace key."""
        if namespace_key:
            return _NS_LABEL.get(namespace_key, namespace_key)
        return "General"

    async def _ensure_hierarchy(self, namespace_key: str | None) -> tuple[KnowledgeNode, KnowledgeNode]:
        """Ensure SUBJECT → TOPIC → SUBTOPIC nodes exist for a namespace.

        Works for ANY arXiv namespace — known ones use curated labels from the
        lookup dicts; unknown ones derive sensible labels from the namespace
        prefix/suffix without ever normalizing the key to a different namespace.
        This ensures the subtopic node always carries the original namespace_key
        so the API namespace filter includes it correctly.

        Returns ``(topic_node, subtopic_node)``.
        """
        # ── Subject root node ─────────────────────────────────────────────────
        # Subjects span namespaces — keyed only by label, namespace_key=None.
        subject_label = _NS_TO_SUBJECT.get(namespace_key) or _derive_subject(namespace_key)
        subject_node = await self._repo.get_or_create_node(
            NodeType.topic,
            label=subject_label,
            namespace_key=None,
        )

        # ── Domain TOPIC node ─────────────────────────────────────────────────
        topic_label = _NS_TO_TOPIC.get(namespace_key) or _derive_topic(namespace_key)
        topic_node = await self._repo.get_or_create_node(
            NodeType.topic,
            label=topic_label,
            namespace_key=None,
        )
        try:
            await self._repo.create_edge(subject_node.id, topic_node.id, EdgeType.has_subtopic)
        except Exception as exc:
            log.debug("_ensure_hierarchy: subject→topic edge already exists — %s", exc)

        # ── Namespace SUBTOPIC node ───────────────────────────────────────────
        # Always use the ORIGINAL namespace_key — never normalize to a different
        # namespace. Normalizing caused mismatches with the API namespace filter.
        subtopic_label = _NS_LABEL.get(namespace_key) or _derive_subtopic_label(namespace_key)
        subtopic_node = await self._repo.get_or_create_node(
            NodeType.subtopic,
            label=subtopic_label,
            namespace_key=namespace_key,
        )
        try:
            await self._repo.create_edge(topic_node.id, subtopic_node.id, EdgeType.has_subtopic)
        except Exception as exc:
            log.debug("_ensure_hierarchy: topic→subtopic edge already exists — %s", exc)
        return topic_node, subtopic_node

    async def add_paper_node(self, paper: Paper) -> None:
        """Register a paper in the knowledge graph under the correct hierarchy.

        Ensures the TOPIC → SUBTOPIC chain exists for the paper's namespace,
        then creates a PAPER node and edges to its CONCEPT and METHOD leaf
        nodes (``introduces`` and ``uses_method`` edge types respectively).
        All node and edge creation is idempotent.

        Args:
            paper: The ``Paper`` ORM object to register. Its ``namespace_key``,
                ``title``, ``key_concepts``, and ``methods_used`` fields are
                used to build the graph structure.
        """
        _, subtopic_node = await self._ensure_hierarchy(paper.namespace_key)

        paper_node = await self._repo.get_or_create_node(
            NodeType.paper,
            label=paper.title,
            namespace_key=paper.namespace_key,
            paper_id=paper.id,
        )

        # Link subtopic → paper
        await self._repo.create_edge(subtopic_node.id, paper_node.id, EdgeType.belongs_to)

        for concept in paper.key_concepts or []:
            concept_node = await self._repo.get_or_create_node(
                NodeType.concept,
                label=concept,
                namespace_key=paper.namespace_key,
            )
            await self._repo.create_edge(paper_node.id, concept_node.id, EdgeType.introduces)

        for method in paper.methods_used or []:
            method_node = await self._repo.get_or_create_node(
                NodeType.method,
                label=method,
                namespace_key=paper.namespace_key,
            )
            await self._repo.create_edge(paper_node.id, method_node.id, EdgeType.uses_method)

        # Invalidate the cached subgraph so the new paper appears on next graph load
        # without requiring a manual Clear or Build Deep.
        await GraphService.clear_subgraph_cache(paper.namespace_key)
        await GraphService.clear_subgraph_cache(None)

    async def rebuild_clusters(self) -> int:
        """Promote shared concepts to cluster nodes between SUBTOPIC and PAPER.

        Concepts appearing in ≥2 papers within the same subtopic get a
        SUBTOPIC → CONCEPT (has_subtopic) edge, making them intermediate cluster
        nodes. Each such paper is then linked CONCEPT → PAPER (belongs_to) via
        its best-matching cluster. Idempotent — create_edge skips existing edges.
        """
        # All SUBTOPIC nodes
        st_res = await self._db.execute(
            select(KnowledgeNode).where(KnowledgeNode.node_type == NodeType.subtopic)
        )
        subtopic_ids = {n.id for n in st_res.scalars()}

        # All belongs_to edges whose source is a SUBTOPIC (SUBTOPIC → PAPER)
        bt_res = await self._db.execute(
            select(KnowledgeEdge).where(KnowledgeEdge.edge_type == EdgeType.belongs_to)
        )
        subtopic_papers: dict = {}
        for e in bt_res.scalars():
            if e.source_id in subtopic_ids:
                subtopic_papers.setdefault(e.source_id, []).append(e.target_id)

        # PAPER → CONCEPT edges
        ce_res = await self._db.execute(
            select(KnowledgeEdge).where(
                KnowledgeEdge.edge_type.in_([EdgeType.introduces, EdgeType.uses_method])
            )
        )
        paper_concepts: dict = {}
        for e in ce_res.scalars():
            paper_concepts.setdefault(e.source_id, []).append(e.target_id)

        cluster_edges = 0

        for subtopic_id, paper_ids in subtopic_papers.items():
            if len(paper_ids) < 2:
                continue

            # Count how many papers reference each concept (deduplicated per paper)
            concept_freq: dict = {}
            for pid in paper_ids:
                seen: set = set()
                for cid in paper_concepts.get(pid, []):
                    if cid not in seen:
                        concept_freq[cid] = concept_freq.get(cid, 0) + 1
                        seen.add(cid)

            n_clusters = min(8, max(2, len(paper_ids) // 2))
            top_clusters = sorted(
                [(cid, freq) for cid, freq in concept_freq.items() if freq >= 2],
                key=lambda x: -x[1],
            )[:n_clusters]

            if not top_clusters:
                continue

            cluster_freq = dict(top_clusters)
            cluster_ids = list(cluster_freq.keys())

            for cid in cluster_ids:
                # Promote concept to cluster: SUBTOPIC → CONCEPT (has_subtopic)
                await self._repo.create_edge(subtopic_id, cid, EdgeType.has_subtopic)
                cluster_edges += 1

            # Link each paper to its highest-frequency matching cluster
            for pid in paper_ids:
                pcset = set(paper_concepts.get(pid, []))
                matches = [(cid, cluster_freq[cid]) for cid in cluster_ids if cid in pcset]
                if matches:
                    best = max(matches, key=lambda x: x[1])[0]
                    await self._repo.create_edge(best, pid, EdgeType.belongs_to)

        await self._db.commit()
        return cluster_edges

    async def rebuild_hierarchy(self) -> int:
        """Backfill TOPIC → SUBTOPIC → PAPER hierarchy for nodes that lack it.

        Finds PAPER nodes that have no incoming belongs_to edge (i.e. not yet
        wired into the hierarchy) and creates the missing chain.
        """
        # All PAPER nodes
        paper_nodes_res = await self._db.execute(
            select(KnowledgeNode).where(KnowledgeNode.node_type == NodeType.paper)
        )
        paper_nodes = list(paper_nodes_res.scalars())

        # All belongs_to edges (subtopic → paper)
        bt_res = await self._db.execute(
            select(KnowledgeEdge).where(KnowledgeEdge.edge_type == EdgeType.belongs_to)
        )
        wired_targets = {e.target_id for e in bt_res.scalars()}

        orphaned = [n for n in paper_nodes if n.id not in wired_targets]
        for node in orphaned:
            _, subtopic_node = await self._ensure_hierarchy(node.namespace_key)
            await self._repo.create_edge(subtopic_node.id, node.id, EdgeType.belongs_to)

        await self._db.commit()
        return len(orphaned)

    async def get_subgraph(self, namespace_key: str | None, depth: int = 2, use_cache: bool = True) -> dict:
        """Return a serializable subgraph suitable for the React Flow frontend.

        Fetches nodes and edges from the repository, enriches PAPER nodes with
        ``tldr``/``abstract`` snippets and source URLs, and annotates CONCEPT
        and METHOD nodes with a reference-count description.

        Args:
            namespace_key: Restrict to nodes belonging to this arXiv-style
                namespace key. Pass ``None`` to return the full graph across
                all namespaces.
            depth: Passed through to the repository layer (reserved for future
                depth-limited traversal). Defaults to ``2``.

        Returns:
            A dict with two keys:
              - ``"nodes"``: list of node dicts with ``id``, ``type``, ``label``,
                ``namespace_key``, ``paper_id``, ``description``, and
                ``source_url``.
              - ``"edges"``: list of edge dicts with ``id``, ``source``,
                ``target``, ``type``, ``weight``, and ``cross_namespace``.
        """
        # Try the cache first (feed scope only — bookmarks are user-specific)
        if use_cache:
            cached = await GraphService.get_cached_subgraph(namespace_key)
            if cached is not None:
                log.debug("get_subgraph: cache hit ns=%s", namespace_key)
                return cached

        nodes, edges = await self._repo.get_subgraph(namespace_key, depth)

        # Fetch paper tldrs for richer node descriptions
        from app.models.paper import Paper as _Paper
        paper_ids = [n.paper_id for n in nodes if n.paper_id]
        paper_desc: dict[str, str] = {}
        paper_urls: dict[str, str] = {}
        if paper_ids:
            res = await self._db.execute(
                select(_Paper.id, _Paper.tldr, _Paper.abstract, _Paper.source_url).where(_Paper.id.in_(paper_ids))
            )
            for row in res:
                # Only use the TL;DR — never fall back to raw abstract for graph tooltips.
                # Abstracts are verbose and make hover unreadable; TL;DR is the right signal.
                if row.tldr:
                    paper_desc[str(row.id)] = row.tldr
                if row.source_url:
                    paper_urls[str(row.id)] = row.source_url

        # Count how many papers reference each concept/method node
        paper_node_ids: set = {n.id for n in nodes if n.node_type == NodeType.paper}
        ref_count: dict = {}
        for e in edges:
            if e.source_id in paper_node_ids:
                ref_count[e.target_id] = ref_count.get(e.target_id, 0) + 1

        def _desc(n) -> str:
            """Return the description string for a graph node."""
            if n.node_type == NodeType.topic:
                return _TOPIC_DESC.get(n.user_label or n.label, "")
            if n.node_type == NodeType.subtopic:
                return _NS_DESC.get(n.namespace_key or "", "")
            if n.node_type == NodeType.paper:
                return paper_desc.get(str(n.paper_id), "") if n.paper_id else ""
            c = ref_count.get(n.id, 0)
            if not c:
                return ""
            plural = "s" if c != 1 else ""
            return f"Referenced in {c} paper{plural}"

        result = {
            "nodes": [
                {
                    "id": str(n.id),
                    "type": n.node_type.value,
                    "label": n.user_label or n.label,
                    "namespace_key": n.namespace_key,
                    "paper_id": str(n.paper_id) if n.paper_id else None,
                    "description": _desc(n),
                    "source_url": paper_urls.get(str(n.paper_id)) if n.paper_id else None,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "id": str(e.id),
                    "source": str(e.source_id),
                    "target": str(e.target_id),
                    "type": e.edge_type.value,
                    "weight": e.weight,
                    "cross_namespace": e.cross_namespace,
                }
                for e in edges
            ],
        }

        # Write through to cache for next request (skip if bookmark-scoped)
        if use_cache:
            await GraphService.set_cached_subgraph(namespace_key, result)

        return result

    async def expand_node(self, node_id: UUID) -> dict:
        """Return serializable immediate neighbors of a single graph node.

        Args:
            node_id: UUID of the ``KnowledgeNode`` to expand.

        Returns:
            A dict with two keys:
              - ``"nodes"``: list of neighbor node dicts with ``id``, ``type``,
                and ``label``.
              - ``"edges"``: list of outgoing edge dicts with ``id``,
                ``source``, ``target``, and ``type``.
        """
        nodes, edges = await self._repo.expand_node(node_id)
        return {
            "nodes": [
                {"id": str(n.id), "type": n.node_type.value, "label": n.user_label or n.label}
                for n in nodes
            ],
            "edges": [
                {"id": str(e.id), "source": str(e.source_id), "target": str(e.target_id),
                 "type": e.edge_type.value}
                for e in edges
            ],
        }

    async def _auto_deduplicate(self) -> int:
        """Merge duplicate nodes sharing the same (label, node_type, namespace_key).

        Must only be called from write endpoints (e.g. POST /graph/deduplicate) where
        the session lifecycle owns a full transaction.  Never call from read paths —
        the commit() here would corrupt a shared read session.
        """
        from sqlalchemy import func, delete as _del, update as _upd

        dup_check = await self._db.execute(
            select(
                KnowledgeNode.label,
                KnowledgeNode.node_type,
                KnowledgeNode.namespace_key,
                func.count(KnowledgeNode.id).label("cnt"),
                func.min(KnowledgeNode.id).label("keep_id"),
            )
            .group_by(KnowledgeNode.label, KnowledgeNode.node_type, KnowledgeNode.namespace_key)
            .having(func.count(KnowledgeNode.id) > 1)
        )
        duplicates = dup_check.fetchall()
        if not duplicates:
            return 0

        merged = 0
        for row in duplicates:
            keep_id = row.keep_id
            victims_res = await self._db.execute(
                select(KnowledgeNode.id).where(
                    KnowledgeNode.label == row.label,
                    KnowledgeNode.node_type == row.node_type,
                    KnowledgeNode.namespace_key == row.namespace_key,
                    KnowledgeNode.id != keep_id,
                )
            )
            for (vid,) in victims_res.fetchall():
                await self._db.execute(_upd(KnowledgeEdge).where(KnowledgeEdge.source_id == vid).values(source_id=keep_id))
                await self._db.execute(_upd(KnowledgeEdge).where(KnowledgeEdge.target_id == vid).values(target_id=keep_id))
                await self._db.execute(_del(KnowledgeEdge).where(KnowledgeEdge.source_id == KnowledgeEdge.target_id))
                await self._db.execute(_del(KnowledgeNode).where(KnowledgeNode.id == vid))
                merged += 1

        try:
            await self._db.commit()
        except Exception as exc:
            await self._db.rollback()
            log.warning("_auto_deduplicate: commit failed, rolled back — %s", exc)
            return 0
        log.info("_auto_deduplicate: merged %d duplicate node(s)", merged)
        return merged

    async def get_related_papers(self, node_id: UUID) -> list[UUID]:
        """Return paper IDs reachable from a concept or method node."""
        _, edges = await self._repo.expand_node(node_id)
        return [e.target_id for e in edges]

    async def _build_related_edges(
        self,
        namespace_key: str | None,
        papers: list,
        sim_threshold: float = 0.62,
        top_k_per_paper: int = 5,
    ) -> int:
        """Create ``related_to`` edges between papers whose abstract embeddings are similar.

        Uses pgvector ANN search (one query per paper) to find the most similar
        papers efficiently — leverages the IVFFlat/HNSW index rather than a
        brute-force O(N²) pairwise loop.  Only adds edges that don't already
        exist.

        Args:
            namespace_key: Restrict both source and target papers to this namespace
                (``None`` = global).
            papers: List of ``Paper`` ORM objects to consider as sources.  Each
                triggers one ANN lookup.
            sim_threshold: Minimum cosine similarity to add an edge.  Defaults to
                ``0.62`` — high enough to avoid spurious links, low enough to
                capture genuine thematic overlaps.
            top_k_per_paper: Maximum number of ``related_to`` edges per source
                paper.  Defaults to 5.

        Returns:
            The number of new ``related_to`` edges created.
        """
        from sqlalchemy import select, text as _text
        from app.models.paper import PaperChunk as _Chunk
        from app.models.graph import KnowledgeNode as _KN, EdgeType as _ET, NodeType as _NT

        def _vec_str(v: list) -> str:
            """Serialize a float vector to PostgreSQL array literal syntax."""
            return f"[{','.join(str(x) for x in v)}]"

        # Get PAPER graph-node lookup: paper_id → node_id
        paper_node_result = await self._db.execute(
            select(_KN.paper_id, _KN.id).where(
                _KN.node_type == _NT.paper,
                _KN.paper_id.isnot(None),
            )
        )
        paper_to_node: dict = {str(row.paper_id): row.id for row in paper_node_result.fetchall()}

        # Find representative chunk (abstract) for each paper
        chunk_result = await self._db.execute(
            select(_Chunk.paper_id, _Chunk.embedding, _Chunk.embedding_dim, _Chunk.embedding_provider)
            .where(
                _Chunk.section_type == "abstract",
                _Chunk.embedding.isnot(None),
            )
        )
        abstract_chunks: dict = {}
        for row in chunk_result.fetchall():
            pid = str(row.paper_id)
            if pid not in abstract_chunks:
                abstract_chunks[pid] = (row.embedding, row.embedding_dim, row.embedding_provider)

        added = 0
        # Pre-build namespace lookup keyed by paper UUID string so cross-namespace
        # detection is O(1) and works even when the target paper is not in `papers`.
        paper_ns_map: dict[str, str | None] = {str(p.id): p.namespace_key for p in papers}

        for paper in papers:
            pid = str(paper.id)
            src_node_id = paper_to_node.get(pid)
            if not src_node_id or pid not in abstract_chunks:
                continue

            vec, dim, provider = abstract_chunks[pid]
            if vec is None:
                continue

            # ANN search for similar abstracts (uses the pgvector index)
            ns_filter = "AND p.namespace_key = :ns" if namespace_key else ""
            ns_params: dict = {"ns": namespace_key} if namespace_key else {}

            sql = _text(f"""
                SELECT p.id AS paper_id, p.namespace_key,
                       1 - (pc.embedding <=> CAST(:vec AS vector)) AS sim
                FROM paper_chunks pc
                JOIN papers p ON p.id = pc.paper_id
                WHERE pc.section_type = 'abstract'
                  AND pc.embedding IS NOT NULL
                  AND pc.embedding_dim = :dim
                  AND pc.embedding_provider = :provider
                  AND p.id != :src_id
                  AND 1 - (pc.embedding <=> CAST(:vec AS vector)) >= :threshold
                  {ns_filter}
                ORDER BY pc.embedding <=> CAST(:vec AS vector)
                LIMIT :top_k
            """)
            try:
                rows = await self._db.execute(sql, {
                    "vec": _vec_str(list(vec)),
                    "dim": dim,
                    "provider": provider.value if hasattr(provider, "value") else str(provider),
                    "src_id": paper.id,
                    "threshold": sim_threshold,
                    "top_k": top_k_per_paper,
                    **ns_params,
                })
                for row in rows.fetchall():
                    tgt_node_id = paper_to_node.get(str(row.paper_id))
                    if not tgt_node_id:
                        continue
                    # namespace_key comes directly from the SQL row — accurate even for
                    # papers outside the `papers` list (cross-namespace targets).
                    cross_ns = paper.namespace_key != row.namespace_key
                    edge = await self._repo.create_edge(
                        src_node_id, tgt_node_id, _ET.related_to,
                        weight=round(float(row.sim), 3),
                        cross_namespace=cross_ns,
                    )
                    if edge:
                        added += 1
            except Exception as exc:
                log.debug("_build_related_edges: ANN query failed for paper=%s: %s", pid, exc)

        log.info("_build_related_edges: added %d related_to edges for %d papers", added, len(papers))
        return added

    # ── Subgraph cache (feed scope only — bookmarks are user-specific) ────────────
    # Keys: "graph:subgraph:feed:{ns_hash}" → {nodes: [...], edges: [...]}
    # Cleared when build_deep_graph completes or clear_graph is called.
    _CACHE_TTL = 14_400  # 4 hours

    @staticmethod
    def _ns_hash(namespace_key: str | None) -> str:
        """Return a short (12-char) SHA-256 hash of a namespace key for use as a cache key suffix."""
        import hashlib
        key = namespace_key or "__all__"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @staticmethod
    async def get_cached_subgraph(namespace_key: str | None) -> dict | None:
        """Return cached feed-scope subgraph or None if cache miss/expired."""
        try:
            from app.adapters.cache import get_cache
            cache = get_cache()
            return await cache.get(f"graph:subgraph:feed:{GraphService._ns_hash(namespace_key)}")
        except Exception:
            return None

    @staticmethod
    async def set_cached_subgraph(namespace_key: str | None, data: dict) -> None:
        """Store subgraph in cache."""
        try:
            from app.adapters.cache import get_cache
            cache = get_cache()
            await cache.set(
                f"graph:subgraph:feed:{GraphService._ns_hash(namespace_key)}",
                data,
                ttl_seconds=GraphService._CACHE_TTL,
            )
        except Exception:
            pass

    @staticmethod
    async def clear_subgraph_cache(namespace_key: str | None = None) -> None:
        """Invalidate cached subgraph for a namespace (or all if namespace_key is None)."""
        try:
            from app.adapters.cache import get_cache
            cache = get_cache()
            if namespace_key:
                await cache.delete(f"graph:subgraph:feed:{GraphService._ns_hash(namespace_key)}")
            else:
                # Clear all — iterate known namespace keys
                for k in [None, "__all__"]:
                    await cache.delete(f"graph:subgraph:feed:{GraphService._ns_hash(k)}")
        except Exception:
            pass

    async def build_deep_graph(self, namespace_key: str | None = None, orientation: str = "both") -> dict:
        """Build a deep, LLM-generated TOPIC → SUBTOPIC → CLUSTER → PAPER hierarchy.

        The LLM taxonomizes papers into research areas (SUBTOPIC nodes) and
        thematic clusters (CONCEPT nodes) based on their abstracts and key
        concepts. The resulting 5-level hierarchy is:
        TOPIC → SUBTOPIC → CONCEPT cluster → PAPER → CONCEPT/METHOD leaf.
        """
        from app.adapters.llm import get_llm_adapter

        # Fetch ALL papers for the namespace (no hard cap — processed in batches).
        # Orders by (novelty + relevance) so most significant papers are taxonomized
        # first when batch count is large; newer/lower-quality papers fill later batches.
        from app.models.paper import Paper as _Paper
        q = (
            select(_Paper)
            .order_by(
                (_Paper.novelty_score + _Paper.relevance_score).desc().nullslast(),
                _Paper.ingested_at.desc(),
            )
        )
        if namespace_key:
            q = q.where(_Paper.namespace_key == namespace_key)
        result = await self._db.execute(q)
        papers = list(result.scalars())

        # Always create the base Subject → Topic → Subtopic hierarchy for this namespace
        # so it appears in the graph even if no papers have been ingested yet.
        # This ensures all 8 selected namespaces show up in the graph as soon as
        # Build Deep is triggered, not just the ones with existing papers.
        try:
            await self._ensure_hierarchy(namespace_key)
            await self._db.commit()
        except Exception as exc:
            log.warning("build_deep_graph: _ensure_hierarchy failed for %s — %s", namespace_key, exc)

        if not papers:
            log.info("build_deep_graph: no papers for %s — hierarchy ensured, returning", namespace_key)
            await GraphService.clear_subgraph_cache(None)
            return {"areas_created": 0, "clusters_created": 0, "papers_mapped": 0, "total_papers_processed": 0}

        cache_key = namespace_key or "__all__"
        if GraphService._build_cache.get(cache_key) == len(papers):
            log.info("build_deep_graph: up to date for %s (%d papers)", namespace_key, len(papers))
            return {"areas_created": 0, "clusters_created": 0, "papers_mapped": 0, "total_papers_processed": len(papers), "already_up_to_date": True}

        llm = get_llm_adapter()

        # Build a lookup: paper uuid → paper object
        id_to_paper: dict[str, _Paper] = {str(p.id): p for p in papers}

        # Format paper summaries for LLM (batch 40 at a time)
        def _summarize(p: _Paper) -> str:
            """Format a paper into a compact single-line string for the LLM batch prompt."""
            concepts = ", ".join((p.key_concepts or [])[:6])
            abstract_snippet = (p.abstract or "")[:220].rstrip()
            return f"PAPER_ID:{p.id}|TITLE:{p.title}|CONCEPTS:{concepts}|ABSTRACT:{abstract_snippet}"

        _BATCH = 40
        batches = [papers[i:i + _BATCH] for i in range(0, len(papers), _BATCH)]

        # Orientation shapes the vocabulary of area and cluster names
        orientation_hint = {
            "research": (
                "Name areas, sub-areas, and clusters using precise academic terminology. "
                "Prefer names that reflect the theoretical contribution or research paradigm "
                "(e.g. 'Attention Mechanisms', 'Contrastive Representation Learning'). "
            ),
            "production": (
                "Name areas, sub-areas, and clusters using applied, practical terminology. "
                "Prefer problem-domain names over technique names "
                "(e.g. 'Language Understanding & Generation', 'Visual Perception Systems'). "
            ),
            "both": "",
        }.get(orientation, "")

        # ── Phase 1: Create a canonical bounded taxonomy from a representative sample ──
        # Using a single canonical structure prevents the "batch explosion" problem where
        # each batch independently invents its own area names, leading to hundreds of
        # duplicate/overlapping areas.  Hard caps: 3-5 areas, 2-4 sub-areas, 2-5 clusters.
        sample_size = min(40, len(papers))
        sample_block = "\n".join(_summarize(p) for p in papers[:sample_size])

        canonical_prompt = (
            "You are a research taxonomy expert. Read these sample papers and define "
            "a canonical 3-level taxonomy that will be used to classify ALL papers in this research area.\n\n"
            "HARD CAPS (do not exceed):\n"
            "  • 3–5 broad research areas\n"
            "  • 2–4 sub-areas per area\n"
            "  • 2–5 specific clusters per sub-area\n\n"
            + orientation_hint
            + "Return ONLY the taxonomy STRUCTURE — no paper assignments, no prose, no markdown.\n"
            'JSON shape: {"areas": {"Area Name": {"Sub-area Name": ["cluster1", "cluster2", ...]}}}\n\n'
            "Names must be concise (2-5 words). Clusters should be specific enough to distinguish "
            "individual research threads (e.g. 'Sparse Attention for Long Contexts' not 'Attention').\n\n"
            f"Sample papers:\n{sample_block}"
        )

        canonical: dict[str, dict[str, list[str]]] = {}
        try:
            canon_res = await llm.complete(
                [{"role": "user", "content": canonical_prompt}],
                model=llm.quality_model,
                temperature=0.1,
                max_tokens=2000,
            )
            raw = canon_res.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rstrip("`").strip()
            canonical = json.loads(raw).get("areas", {})
            log.info(
                "build_deep_graph phase1: %d areas, %d total clusters",
                len(canonical),
                sum(len(clusters) for subareas in canonical.values() for clusters in subareas.values()),
            )
        except Exception as exc:
            log.warning("build_deep_graph: phase-1 canonical taxonomy failed — %s", exc)
            # Fallback: single-phase (old approach) if canonical fails
            canonical = {}

        # ── Phase 2: Assign all papers to the canonical taxonomy in batches ─────────
        # Structure: {area: {sub_area: {cluster: [paper_uuids]}}}
        merged: dict[str, dict[str, dict[str, list[str]]]] = {}

        if canonical:
            # Two-phase: assign papers to fixed canonical structure
            canonical_json = json.dumps(canonical, indent=2)
            assign_prompt = (
                "Assign each paper to the BEST matching cluster in this canonical taxonomy.\n\n"
                f"Taxonomy:\n{canonical_json}\n\n"
                "Rules:\n"
                "  • Use EXACT area/sub-area/cluster names from the taxonomy.\n"
                "  • Every paper must be assigned to exactly one cluster.\n"
                "  • Choose the most specific matching cluster.\n\n"
                "Return ONLY valid JSON — no prose, no markdown:\n"
                '{"assignments": [{"paper_id": "uuid", "area": "...", "sub_area": "...", "cluster": "..."}]}'
            )

            for batch in batches:
                paper_block = "\n".join(_summarize(p) for p in batch)
                try:
                    res = await llm.complete(
                        [
                            {"role": "system", "content": assign_prompt},
                            {"role": "user", "content": f"Papers to assign:\n{paper_block}"},
                        ],
                        model=llm.cheap_model,  # assignment is simpler → use cheap model
                        temperature=0.0,
                        max_tokens=3000,
                    )
                    raw = res.text.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                        raw = raw.rstrip("`").strip()
                    assignments: list[dict] = json.loads(raw).get("assignments", [])
                    for a in assignments:
                        area = a.get("area", "")
                        sub_area = a.get("sub_area", "")
                        cluster = a.get("cluster", "")
                        paper_id = a.get("paper_id", "")
                        if not all([area, sub_area, cluster, paper_id]):
                            continue
                        merged.setdefault(area, {}).setdefault(sub_area, {}).setdefault(cluster, []).append(paper_id)
                except Exception as exc:
                    log.warning("build_deep_graph phase2 batch failed: %s", exc)
                    continue
        else:
            # Fallback single-phase (old approach) — only used when phase-1 fails
            system_prompt = (
                "You are a research taxonomy expert. Organize these papers into a 3-level taxonomy.\n"
                "HARD CAPS: 3-5 areas, 2-4 sub-areas per area, 2-5 clusters per sub-area.\n"
                "Every paper assigned to exactly one cluster. Names: 2-5 words.\n"
                + orientation_hint
                + 'Return ONLY: {"taxonomy": {"Area": {"Sub-area": {"Cluster": ["uuid", ...]}}}}'
            )
            for batch in batches:
                paper_block = "\n".join(_summarize(p) for p in batch)
                try:
                    res = await llm.complete(
                        [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": f"Papers:\n{paper_block}"}],
                        model=llm.quality_model, temperature=0.2, max_tokens=4000,
                    )
                    raw = res.text.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    taxonomy_data = json.loads(raw).get("taxonomy", {})
                except Exception as exc:
                    log.warning("build_deep_graph: fallback batch failed — %s", exc)
                    continue
                for area, sub_areas in taxonomy_data.items():
                    for sub_area, clusters in (sub_areas.items() if isinstance(sub_areas, dict) else {}.items()):
                        if isinstance(clusters, dict):
                            for cluster, uuids in clusters.items():
                                if isinstance(uuids, list):
                                    merged.setdefault(area, {}).setdefault(sub_area, {}).setdefault(cluster, []).extend(uuids)
                        elif isinstance(clusters, list):
                            merged.setdefault(area, {}).setdefault("General", {}).setdefault(sub_area, []).extend(clusters)

        if not merged:
            log.warning("build_deep_graph: LLM returned no taxonomy")
            return {"areas_created": 0, "sub_areas_created": 0, "clusters_created": 0, "papers_mapped": 0, "total_papers_processed": len(papers)}

        # Ensure base TOPIC → SUBTOPIC hierarchy
        _, subtopic_node = await self._ensure_hierarchy(namespace_key)

        areas_created = 0
        sub_areas_created = 0
        clusters_created = 0
        papers_mapped = 0

        for area_name, sub_areas in merged.items():
            # TOPIC → SUBTOPIC → CONCEPT(area)
            area_node = await self._repo.get_or_create_node(
                NodeType.concept, label=area_name, namespace_key=namespace_key,
            )
            await self._repo.create_edge(subtopic_node.id, area_node.id, EdgeType.has_subtopic)
            areas_created += 1

            for sub_area_name, clusters in sub_areas.items():
                # area → CONCEPT(sub-area) — the new intermediate level
                sub_area_node = await self._repo.get_or_create_node(
                    NodeType.concept, label=sub_area_name, namespace_key=namespace_key,
                )
                await self._repo.create_edge(area_node.id, sub_area_node.id, EdgeType.has_subtopic)
                sub_areas_created += 1

                for cluster_name, paper_uuids in clusters.items():
                    # sub-area → CONCEPT(cluster)
                    cluster_node = await self._repo.get_or_create_node(
                        NodeType.concept, label=cluster_name, namespace_key=namespace_key,
                    )
                    await self._repo.create_edge(sub_area_node.id, cluster_node.id, EdgeType.has_subtopic)
                    clusters_created += 1

                    for uuid_str in paper_uuids:
                        paper = id_to_paper.get(uuid_str)
                        if not paper:
                            continue
                        paper_node = await self._repo.get_or_create_node(
                            NodeType.paper, label=paper.title,
                            namespace_key=paper.namespace_key, paper_id=paper.id,
                        )
                        await self._repo.create_edge(cluster_node.id, paper_node.id, EdgeType.belongs_to)

                        # Leaf nodes: key concepts + methods under each paper
                        for concept in (paper.key_concepts or [])[:6]:
                            concept_node = await self._repo.get_or_create_node(
                                NodeType.concept, label=concept, namespace_key=paper.namespace_key,
                            )
                            await self._repo.create_edge(paper_node.id, concept_node.id, EdgeType.introduces)
                        for method in (paper.methods_used or [])[:4]:
                            method_node = await self._repo.get_or_create_node(
                                NodeType.method, label=method, namespace_key=paper.namespace_key,
                            )
                            await self._repo.create_edge(paper_node.id, method_node.id, EdgeType.uses_method)

                        papers_mapped += 1

            # Commit each area incrementally so partial progress is visible in the
            # graph while the rest of the namespace is still being processed.
            await self._db.commit()
            await GraphService.clear_subgraph_cache(namespace_key)
            await GraphService.clear_subgraph_cache(None)

        # Add cross-paper related_to edges using stored embeddings
        related_edges = await self._build_related_edges(namespace_key, papers)

        await self._db.commit()

        # Invalidate both the per-namespace cache AND the global cache.
        # The graph page always calls get_subgraph(None) so its cache key is
        # _ns_hash(None) ("__all__") — clearing only the per-namespace key is insufficient.
        await GraphService.clear_subgraph_cache(namespace_key)
        await GraphService.clear_subgraph_cache(None)  # clears the global feed cache

        log.info(
            "build_deep_graph: areas=%d sub_areas=%d clusters=%d papers=%d related_edges=%d total=%d",
            areas_created, sub_areas_created, clusters_created, papers_mapped, related_edges, len(papers),
        )
        GraphService._build_cache[cache_key] = len(papers)
        return {
            "areas_created": areas_created,
            "sub_areas_created": sub_areas_created,
            "clusters_created": clusters_created,
            "papers_mapped": papers_mapped,
            "related_edges_added": related_edges,
            "total_papers_processed": len(papers),
        }
