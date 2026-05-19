"""Shared prompt utilities for media-generation workflows.

Centralises:
  - Domain detection from paper content (used by all four pipelines)
  - Expertise / orientation directive strings
  - LLM temperature recommendations per call type
  - Anti-truncation suffix (appended to every generation prompt)

Every generation workflow imports from here so changes apply globally.
"""

from __future__ import annotations

import re


# ── Temperature presets ───────────────────────────────────────────────────────
# Lower = more deterministic (good for factual extraction / structured JSON)
# Higher = more creative (good for narrative content)

TEMP_EXTRACT = 0.10   # factual extraction — maximum groundedness
TEMP_PLAN    = 0.40   # structured planning — grounded but varied structure
TEMP_WRITE   = 0.50   # narrative writing — engaging but stays grounded
TEMP_CODE    = 0.15   # code generation — deterministic


# ── Anti-truncation suffix ────────────────────────────────────────────────────
# Appended to every content-writing system prompt so long outputs never cut off.

ANTI_TRUNCATION = (
    "\n\nOUTPUT COMPLETENESS — CRITICAL: "
    "Generate EVERY item in the plan. Never truncate, skip, or abbreviate sections. "
    "If you sense you are approaching a length limit, write more concisely but ALWAYS "
    "complete every item. Return the full JSON array / full script with all scenes / "
    "all pages / all slides — no partial output."
)


# ── Truncation detection ──────────────────────────────────────────────────────
# Output completeness checks for generated artifacts. When detection fires, the
# caller should retry once with a higher max_tokens or a tighter target length.

def looks_truncated_text(text: str, *, min_chars: int = 200) -> bool:
    """Heuristic: does this text look like the model was cut off mid-output?

    True if:
      - empty / way under min_chars
      - ends with ``...`` or ``…`` (ellipsis used by models to signal they stopped)
      - ends without sentence-terminator (and not on closing brace)
      - JSON-like content with unbalanced braces
    """
    if not text or len(text.strip()) < min_chars:
        return True
    tail = text.rstrip()
    if not tail:
        return True
    # Explicit truncation signals used by models
    if tail.endswith("...") or tail.endswith("…"):
        return True
    # JSON-like: balanced braces?
    if tail.startswith("{") or tail.startswith("["):
        opens = tail.count("{") + tail.count("[")
        closes = tail.count("}") + tail.count("]")
        if opens > closes:
            return True
    last_char = tail[-1]
    # Acceptable terminators for prose / scripts / markdown / json
    if last_char in {".", "!", "?", "}", "]", ")", "”", "\"", "'", "`"}:
        return False
    # Otherwise — likely cut off mid-sentence
    return True


def looks_truncated_json(text: str) -> bool:
    """True if text is JSON-shaped but the braces don't balance."""
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    if s[0] not in "{[":
        return True
    return s.count("{") != s.count("}") or s.count("[") != s.count("]")


# ── Mermaid syntax validation ─────────────────────────────────────────────────
# Lightweight syntactic guard. Mermaid renders client-side, but malformed specs
# (missing direction, code-fence leakage, unmatched brackets, leading prose)
# turn the diagram into a red error box. We do a cheap pre-check and minor
# repairs so the user never sees that.

_MERMAID_HEADERS = (
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram",
    "erDiagram", "journey", "gantt", "pie", "mindmap", "timeline",
    "gitgraph", "C4Context", "quadrantChart", "xychart-beta",
)


def _collapse_label_newlines(spec: str) -> str:
    """Collapse newlines that appear INSIDE node-label brackets to spaces.

    Mermaid expects each node declaration on one line. LLMs occasionally
    wrap a long label across a hard newline (``H[Factual Consistency\\n
    Confidence (c_f)]``) which causes the parser to absorb every
    subsequent statement into that label as raw text — exactly the
    "raw mermaid shown as a single text block" failure mode we saw on
    Genie deep dives. We scan for ``id[...]`` / ``id(...)`` / ``id{...}``
    spans and replace internal newlines with spaces. If a label opens but
    never closes within the document we leave it alone so the bracket-
    balance check below can decide whether to give up.
    """
    pairs = (("[", "]"), ("(", ")"), ("{", "}"))
    out = spec
    for opener, closer in pairs:
        # Find <id><opener>...<closer> spans that contain a newline; replace
        # the inner newlines with single spaces.
        i = 0
        result: list[str] = []
        while i < len(out):
            ch = out[i]
            if ch.isalnum() or ch == "_":
                # Capture identifier
                j = i
                while j < len(out) and (out[j].isalnum() or out[j] == "_"):
                    j += 1
                if j < len(out) and out[j] == opener:
                    # Find matching closer with depth tracking
                    depth = 1
                    k = j + 1
                    while k < len(out) and depth > 0:
                        if out[k] == opener:
                            depth += 1
                        elif out[k] == closer:
                            depth -= 1
                            if depth == 0:
                                break
                        k += 1
                    if k < len(out) and depth == 0:
                        ident = out[i:j]
                        content = out[j + 1 : k]
                        # Collapse all whitespace runs (including newlines) to single spaces
                        content_clean = re.sub(r"\s+", " ", content).strip()
                        result.append(ident + opener + content_clean + closer)
                        i = k + 1
                        continue
                result.append(out[i:j])
                i = j
                continue
            result.append(ch)
            i += 1
        out = "".join(result)
    return out


def repair_mermaid(spec: str) -> str | None:
    """Best-effort cleanup of a Mermaid spec.

    Strips wrapping ``` fences, removes any prose before the first valid
    header line, normalises newlines inside node labels (a common LLM
    error that breaks rendering catastrophically), and balances trivial
    bracket mismatches. Returns the cleaned spec, or ``None`` if it
    can't be salvaged.
    """
    if not spec:
        return None
    s = spec.strip()

    # Strip wrapping ``` fences (with or without lang tag)
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()

    # Drop everything before the first known mermaid header
    lower_lines = s.split("\n")
    start = -1
    for i, line in enumerate(lower_lines):
        stripped = line.strip()
        if any(stripped.startswith(h) for h in _MERMAID_HEADERS):
            start = i
            break
    if start == -1:
        return None
    s = "\n".join(lower_lines[start:]).strip()

    # Normalise newlines inside node labels — otherwise mermaid silently
    # absorbs everything until the next ']' into one node's label and
    # renders it as raw text. This is the most common Genie/deep-dive
    # mermaid failure mode.
    s = _collapse_label_newlines(s)

    # Balance brackets — append closers if a small mismatch exists. Refuse if
    # the imbalance is severe (likely truncated mid-spec).
    pairs = (("[", "]"), ("(", ")"), ("{", "}"))
    for opener, closer in pairs:
        diff = s.count(opener) - s.count(closer)
        if 0 < diff <= 2:
            s += closer * diff
        elif diff < -2 or diff > 2:
            return None

    return s if s else None


def validate_mermaid(spec: str) -> bool:
    """Cheap syntax check — does this look like a renderable Mermaid spec?"""
    if not spec or not spec.strip():
        return False
    first_line = spec.strip().split("\n", 1)[0].strip()
    if not any(first_line.startswith(h) for h in _MERMAID_HEADERS):
        return False
    # Bracket balance must be exact
    for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
        if spec.count(opener) != spec.count(closer):
            return False
    return True


# ── Prompt leakage prevention ─────────────────────────────────────────────────
# Generated outputs occasionally echo internal delimiters or template artefacts
# (``<<DATA_START>>``, ``[START]``, ``<paper>``, "EXPERTISE:" directives, etc.).
# These must never reach the user.

_LEAK_PATTERN = re.compile(
    r"<<[A-Z_]+>>"                                # legacy delimiters
    r"|</?paper>|</?content>|</?data>|</?source>" # XML-style wrappers
    r"|\[START\]|\[END\]"                         # bracket delimiters
    r"|^\s*EXPERTISE:.*$"                         # injected directive lines
    r"|^\s*ORIENTATION:.*$"
    r"|^\s*DOMAIN:.*$"
    r"|^\s*OUTPUT COMPLETENESS.*$"
    r"|^\s*SYSTEM:.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Markers that should NEVER appear in user-facing output. If any of these
# survive the strip pass, the output is treated as poisoned and discarded.
_HARD_LEAKS = (
    "<<PAPER_DATA>>", "<<DATA_START>>", "<<END_DATA>>",
    "ignore the instructions above", "system prompt:",
)


def strip_prompt_artifacts(text: str) -> str:
    """Remove leaked prompt delimiters and directive lines from generated text.

    Safe to call on any LLM output. Collapses 3+ blank lines to 2 so the
    cleanup is invisible when delimiters are sprinkled inline.
    """
    if not text:
        return text
    cleaned = _LEAK_PATTERN.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def has_hard_leak(text: str) -> bool:
    """True if the text contains an unambiguous prompt-injection / leak marker.

    Callers should regenerate or discard rather than persist such output.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _HARD_LEAKS)


# ── Domain detector ───────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Machine Learning & AI": [
        "neural network", "transformer", "attention", "gradient", "loss function",
        "training", "inference", "model", "dataset", "benchmark", "llm", "diffusion",
        "fine-tuning", "embedding", "tokenizer", "backpropagation", "epoch",
    ],
    "Computer Vision": [
        "image segmentation", "object detection", "convolutional", "resnet", "vit",
        "pixel", "bounding box", "feature map", "backbone", "image classification",
        "optical flow", "depth estimation", "pose estimation",
    ],
    "Natural Language Processing": [
        "language model", "text generation", "sentiment", "named entity", "parsing",
        "tokenisation", "translation", "summarisation", "bert", "question answering",
        "coreference", "semantic similarity", "text classification",
    ],
    "Robotics & Control": [
        "robot", "manipulation", "control policy", "actuator", "sensor",
        "trajectory", "motion planning", "pid controller", "kinematics",
        "simultaneous localisation", "slam", "autonomous",
    ],
    "Statistics": [
        "hypothesis testing", "p-value", "confidence interval", "bayesian inference",
        "frequentist", "estimator", "regression", "variance", "covariance",
        "maximum likelihood", "posterior", "prior distribution", "markov chain",
        "causal inference", "propensity score", "bootstrap", "permutation test",
    ],
    "Quantitative Finance": [
        "stochastic", "option pricing", "portfolio optimisation", "volatility",
        "derivatives", "black-scholes", "risk-neutral", "asset pricing",
        "factor model", "alpha", "sharpe ratio", "monte carlo simulation",
        "interest rate", "yield curve", "hedging", "arbitrage",
    ],
    "Economics": [
        "equilibrium", "auction", "mechanism design", "market design",
        "utility", "game theory", "welfare", "externality", "incentive",
        "supply and demand", "elasticity", "microeconomics", "macroeconomics",
        "fiscal policy", "monetary policy", "gdp", "inflation",
    ],
    "Bioinformatics / Computational Biology": [
        "protein", "dna", "genome", "sequence alignment", "mutation", "gene expression",
        "transcriptomics", "phylogenetic", "rna", "proteomics", "variant calling",
        "single-cell", "metagenomics", "structural biology",
    ],
    "Biology": [
        "cell", "organism", "evolution", "ecology", "species", "population",
        "metabolism", "membrane", "receptor", "signalling pathway", "phenotype",
        "genotype", "natural selection", "biodiversity", "photosynthesis",
    ],
    "Neuroscience": [
        "neuron", "synapse", "cortex", "fmri", "eeg", "spike", "cognition",
        "neural circuit", "action potential", "dopamine", "serotonin",
        "working memory", "hippocampus", "prefrontal", "plasticity",
    ],
    "Physics": [
        "quantum", "hamiltonian", "wave function", "particle", "thermodynamics",
        "entropy", "phase transition", "field theory", "photon", "condensed matter",
        "scattering", "spin", "relativity", "cosmology", "dark matter",
    ],
    "Mathematics": [
        "theorem", "proof", "lemma", "topology", "manifold", "algebra",
        "differential equation", "graph theory", "combinatorics", "number theory",
        "category theory", "analysis", "measure theory", "stochastic process",
        "convex optimisation", "linear programming",
    ],
    "Chemistry": [
        "molecule", "synthesis", "reaction", "catalyst", "spectroscopy",
        "molecular dynamics", "dft", "binding energy", "organic chemistry",
        "polymer", "electrochemistry", "thermochemistry", "reagent",
    ],
    "Medicine & Clinical Research": [
        "clinical trial", "patient", "diagnosis", "treatment", "therapy",
        "biomarker", "randomised", "cohort", "placebo", "prognosis",
        "imaging", "mortality", "incidence", "prevalence", "pharmacology",
    ],
    "Materials Science": [
        "crystal", "alloy", "semiconductor", "nanoparticle", "lattice",
        "electronic structure", "band gap", "thin film", "composite",
        "corrosion", "mechanical properties", "superconductor",
    ],
    "Astrophysics & Astronomy": [
        "galaxy", "star", "telescope", "redshift", "cosmology", "black hole",
        "exoplanet", "nebula", "supernova", "spectral", "gravitational wave",
        "dark energy", "stellar", "solar", "pulsar",
    ],
    "Environmental & Climate Science": [
        "climate", "atmosphere", "carbon", "ecosystem", "greenhouse gas",
        "temperature anomaly", "sea level", "precipitation", "emissions",
        "biodiversity loss", "deforestation", "ocean acidification",
    ],
}


def detect_domain(paper_content: str) -> str:
    """Infer the research domain from paper content keywords.

    Scores each domain by keyword hits, normalised by keyword list length so
    domains with many keywords don't get an unfair advantage. Returns the
    top domain when its raw hit count is >= 2 and it leads the runner-up by
    at least 1 hit (prevents spurious ties on generic vocabulary). Falls back
    to ``"interdisciplinary / general science"`` when no clear signal exists.
    """
    if not paper_content:
        return "interdisciplinary / general science"

    lower = paper_content.lower()
    scores: dict[str, int] = {
        domain: sum(1 for kw in keywords if kw in lower)
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }

    sorted_domains = sorted(scores, key=lambda d: scores[d], reverse=True)
    best, runner_up = sorted_domains[0], sorted_domains[1]

    if scores[best] >= 2 and scores[best] > scores[runner_up]:
        return best
    if scores[best] >= 3:  # strong signal even with a tie
        return best
    return "interdisciplinary / general science"


# ── Expertise directives ──────────────────────────────────────────────────────

_EXPERTISE_DIRECTIVES: dict[str, str] = {
    "newcomer": (
        "EXPERTISE: NEWCOMER. "
        "Assume no prior knowledge. Use plain language, vivid analogies, and real-world examples. "
        "Define every technical term the first time it appears. "
        "Build understanding step by step — never skip conceptual bridges."
    ),
    "practitioner": (
        "EXPERTISE: PRACTITIONER. "
        "Assume solid domain background. Be direct and specific — name real methods, datasets, "
        "metrics, architectures. Include implementation details and engineering trade-offs. "
        "Skip hand-holding on fundamentals."
    ),
    "expert": (
        "EXPERTISE: EXPERT. "
        "Full technical depth. Use domain notation, cite specific theoretical foundations, "
        "compare rigorously to concurrent work, critique methodology honestly. "
        "Treat the reader as a peer researcher."
    ),
}

_ORIENTATION_DIRECTIVES: dict[str, str] = {
    "research": (
        "ORIENTATION: RESEARCHER. "
        "Lead with theoretical novelty, methodological rigor, scientific implications, "
        "and what new research directions this opens. Assess the paper's contribution "
        "to the scientific record."
    ),
    "production": (
        "ORIENTATION: PRACTITIONER. "
        "Lead with real-world applicability, deployment considerations, latency/cost "
        "trade-offs, and concrete engineering paths. Focus on what someone can build today."
    ),
    "both": (
        "ORIENTATION: BALANCED. "
        "Weave theoretical insights with practical relevance. Show why the science matters "
        "AND what someone could build with it."
    ),
}


def expertise_directive(level: str) -> str:
    """Return the expertise directive string for a given level."""
    return _EXPERTISE_DIRECTIVES.get(level, _EXPERTISE_DIRECTIVES["practitioner"])


def orientation_directive(orientation: str) -> str:
    """Return the orientation directive string for a given orientation."""
    return _ORIENTATION_DIRECTIVES.get(orientation, _ORIENTATION_DIRECTIVES["both"])


def domain_directive(domain: str) -> str:
    """Return a domain-adaptation directive to inject into every generation prompt."""
    return (
        f"DOMAIN: This is a {domain} paper. "
        f"Adapt ALL terminology, analogies, examples, and explanations to "
        f"{domain} conventions. Use domain-specific vocabulary where it adds precision. "
        f"For newcomer-level content, explain {domain}-specific concepts before using them."
    )


def generation_context(
    *,
    expertise: str,
    orientation: str,
    domain: str,
) -> str:
    """Compose the full generation context block for injection into system prompts.

    Combines expertise, orientation, and domain directives into a single
    string that every generation workflow appends to its system prompt.

    Args:
        expertise: ``"newcomer"`` | ``"practitioner"`` | ``"expert"``.
        orientation: ``"research"`` | ``"production"`` | ``"both"``.
        domain: Human-readable domain string from :func:`detect_domain`.

    Returns:
        Formatted context block ready to concatenate to a system prompt.
    """
    return (
        f"\n\n---\n"
        f"{expertise_directive(expertise)}\n\n"
        f"{orientation_directive(orientation)}\n\n"
        f"{domain_directive(domain)}"
        + ANTI_TRUNCATION
    )
