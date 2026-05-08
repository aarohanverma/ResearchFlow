"""NamespaceManager — authoritative mapping from topic selection to namespace_key.

namespace_key format follows arXiv category identifiers directly.
e.g. cs.AI, cs.LG, quant-ph, cond-mat.stat-mech
"""

# Canonical namespace_key → arXiv RSS category mapping.
# For new arXiv-aligned keys the mapping is identity (key == arXiv category).
# Legacy custom keys are kept for backward compat.
NAMESPACE_TO_ARXIV: dict[str, str] = {
    # ── Computer Science ───────────────────────────────────────────────────────
    "cs.AI":  "cs.AI",
    "cs.AR":  "cs.AR",
    "cs.CC":  "cs.CC",
    "cs.CE":  "cs.CE",
    "cs.CG":  "cs.CG",
    "cs.CL":  "cs.CL",   # Computation & Language (NLP)
    "cs.CR":  "cs.CR",
    "cs.CV":  "cs.CV",
    "cs.CY":  "cs.CY",
    "cs.DB":  "cs.DB",
    "cs.DC":  "cs.DC",
    "cs.DL":  "cs.DL",
    "cs.DM":  "cs.DM",
    "cs.DS":  "cs.DS",
    "cs.ET":  "cs.ET",
    "cs.FL":  "cs.FL",
    "cs.GR":  "cs.GR",
    "cs.GT":  "cs.GT",
    "cs.HC":  "cs.HC",
    "cs.IR":  "cs.IR",
    "cs.IT":  "cs.IT",
    "cs.LG":  "cs.LG",   # Machine Learning
    "cs.LO":  "cs.LO",
    "cs.MA":  "cs.MA",
    "cs.MS":  "cs.MS",
    "cs.NA":  "cs.NA",
    "cs.NE":  "cs.NE",
    "cs.NI":  "cs.NI",
    "cs.OS":  "cs.OS",
    "cs.PF":  "cs.PF",
    "cs.PL":  "cs.PL",
    "cs.RO":  "cs.RO",
    "cs.SE":  "cs.SE",
    "cs.SI":  "cs.SI",
    "cs.SY":  "cs.SY",
    # ── Physics ────────────────────────────────────────────────────────────────
    "quant-ph":           "quant-ph",
    "gr-qc":              "gr-qc",
    "hep-th":             "hep-th",
    "hep-ph":             "hep-ph",
    "hep-ex":             "hep-ex",
    "hep-lat":            "hep-lat",
    "math-ph":            "math-ph",
    "nucl-th":            "nucl-th",
    "nucl-ex":            "nucl-ex",
    "astro-ph.CO":        "astro-ph.CO",
    "astro-ph.EP":        "astro-ph.EP",
    "astro-ph.GA":        "astro-ph.GA",
    "astro-ph.HE":        "astro-ph.HE",
    "astro-ph.IM":        "astro-ph.IM",
    "astro-ph.SR":        "astro-ph.SR",
    "cond-mat.dis-nn":    "cond-mat.dis-nn",
    "cond-mat.mes-hall":  "cond-mat.mes-hall",
    "cond-mat.mtrl-sci":  "cond-mat.mtrl-sci",
    "cond-mat.other":     "cond-mat.other",
    "cond-mat.quant-gas": "cond-mat.quant-gas",
    "cond-mat.soft":      "cond-mat.soft",
    "cond-mat.stat-mech": "cond-mat.stat-mech",
    "cond-mat.str-el":    "cond-mat.str-el",
    "cond-mat.supr-con":  "cond-mat.supr-con",
    "nlin.AO":            "nlin.AO",
    "nlin.CD":            "nlin.CD",
    "nlin.CG":            "nlin.CG",
    "nlin.PS":            "nlin.PS",
    "nlin.SI":            "nlin.SI",
    "physics.acc-ph":     "physics.acc-ph",
    "physics.ao-ph":      "physics.ao-ph",
    "physics.app-ph":     "physics.app-ph",
    "physics.atom-ph":    "physics.atom-ph",
    "physics.bio-ph":     "physics.bio-ph",
    "physics.chem-ph":    "physics.chem-ph",
    "physics.class-ph":   "physics.class-ph",
    "physics.comp-ph":    "physics.comp-ph",
    "physics.data-an":    "physics.data-an",
    "physics.flu-dyn":    "physics.flu-dyn",
    "physics.gen-ph":     "physics.gen-ph",
    "physics.geo-ph":     "physics.geo-ph",
    "physics.ins-det":    "physics.ins-det",
    "physics.med-ph":     "physics.med-ph",
    "physics.optics":     "physics.optics",
    "physics.plasm-ph":   "physics.plasm-ph",
    "physics.soc-ph":     "physics.soc-ph",
    "physics.space-ph":   "physics.space-ph",
    # ── Mathematics ────────────────────────────────────────────────────────────
    "math.AC": "math.AC",
    "math.AG": "math.AG",
    "math.AP": "math.AP",
    "math.AT": "math.AT",
    "math.CA": "math.CA",
    "math.CO": "math.CO",
    "math.CT": "math.CT",
    "math.CV": "math.CV",
    "math.DG": "math.DG",
    "math.DS": "math.DS",
    "math.FA": "math.FA",
    "math.GM": "math.GM",
    "math.GN": "math.GN",
    "math.GR": "math.GR",
    "math.GT": "math.GT",
    "math.HO": "math.HO",
    "math.IT": "math.IT",
    "math.KT": "math.KT",
    "math.LO": "math.LO",
    "math.MG": "math.MG",
    "math.MP": "math.MP",
    "math.NA": "math.NA",
    "math.NT": "math.NT",
    "math.OA": "math.OA",
    "math.OC": "math.OC",
    "math.PR": "math.PR",
    "math.QA": "math.QA",
    "math.RA": "math.RA",
    "math.RT": "math.RT",
    "math.SG": "math.SG",
    "math.SP": "math.SP",
    "math.ST": "math.ST",
    # ── Statistics ─────────────────────────────────────────────────────────────
    "stat.AP": "stat.AP",
    "stat.CO": "stat.CO",
    "stat.ME": "stat.ME",
    "stat.ML": "stat.ML",
    "stat.OT": "stat.OT",
    "stat.TH": "stat.TH",
    # ── Quantitative Biology ───────────────────────────────────────────────────
    "q-bio.BM": "q-bio.BM",
    "q-bio.CB": "q-bio.CB",
    "q-bio.GN": "q-bio.GN",
    "q-bio.MN": "q-bio.MN",
    "q-bio.NC": "q-bio.NC",
    "q-bio.OT": "q-bio.OT",
    "q-bio.PE": "q-bio.PE",
    "q-bio.QM": "q-bio.QM",
    "q-bio.SC": "q-bio.SC",
    "q-bio.TO": "q-bio.TO",
    # ── EESS ───────────────────────────────────────────────────────────────────
    "eess.AS": "eess.AS",
    "eess.IV": "eess.IV",
    "eess.SP": "eess.SP",
    "eess.SY": "eess.SY",
    # ── Economics ──────────────────────────────────────────────────────────────
    "econ.EM": "econ.EM",
    "econ.GN": "econ.GN",
    "econ.TH": "econ.TH",
    # ── Quantitative Finance ───────────────────────────────────────────────────
    "q-fin.CP": "q-fin.CP",
    "q-fin.EC": "q-fin.EC",
    "q-fin.GN": "q-fin.GN",
    "q-fin.MF": "q-fin.MF",
    "q-fin.PM": "q-fin.PM",
    "q-fin.PR": "q-fin.PR",
    "q-fin.RM": "q-fin.RM",
    "q-fin.ST": "q-fin.ST",
    "q-fin.TR": "q-fin.TR",
    # ── Legacy custom keys (backward compat) ───────────────────────────────────
    "cs.ML":         "cs.LG",
    "cs.NLP":        "cs.CL",
    "cs.agents":     "cs.MA",
    "cs.systems":    "cs.DC",
    "cs.security":   "cs.CR",
    "physics.quantum": "quant-ph",
    "physics.hep":   "hep-ph",
    "math.opt":      "math.OC",
    "math.prob":     "math.PR",
    "bio.genomics":  "q-bio.GN",
    "neuro.comp":    "q-bio.NC",
    "econ.theory":   "econ.TH",
}


# Human-readable subject group labels
_SUBJECT_GROUPS: dict[str, str] = {
    "cs":      "Computer Science",
    "quant":   "Physics",
    "gr":      "Physics",
    "hep":     "Physics",
    "math-ph": "Physics",
    "nucl":    "Physics",
    "astro":   "Physics",
    "cond":    "Physics",
    "nlin":    "Physics",
    "physics": "Physics",
    "math":    "Mathematics",
    "stat":    "Statistics",
    "q-bio":   "Quantitative Biology",
    "eess":    "Electrical Engineering & Systems Science",
    "econ":    "Economics",
    "q-fin":   "Quantitative Finance",
}

# Namespace key label overrides for readability
_NS_LABELS: dict[str, str] = {
    "cs.AI":  "Artificial Intelligence",
    "cs.LG":  "Machine Learning",
    "cs.CL":  "NLP / Computation & Language",
    "cs.CV":  "Computer Vision",
    "cs.RO":  "Robotics",
    "cs.CR":  "Cryptography & Security",
    "cs.SE":  "Software Engineering",
    "cs.DB":  "Databases",
    "cs.DC":  "Distributed Computing",
    "cs.NE":  "Neural & Evolutionary Computing",
    "cs.IR":  "Information Retrieval",
    "cs.GT":  "Game Theory",
    "cs.HC":  "Human-Computer Interaction",
    "cs.MA":  "Multiagent Systems",
    "quant-ph": "Quantum Physics",
    "stat.ML":  "Statistical Machine Learning",
    "q-bio.NC": "Computational Neuroscience",
}


def _group_prefix(key: str) -> str:
    """Return the group prefix used to look up _SUBJECT_GROUPS."""
    prefix = key.split(".")[0].split("-")[0]
    # Handle compound prefixes like astro-ph, cond-mat, q-bio, q-fin, math-ph
    for compound in ("quant-ph", "gr-qc", "math-ph", "astro-ph", "cond-mat", "nlin"):
        if key.startswith(compound):
            return compound.split("-")[0]
    return prefix


class NamespaceManager:
    """Maps application namespace keys to arXiv category identifiers and UI labels.

    Provides helpers for resolving namespace keys to arXiv RSS categories,
    constructing keys from subject/topic pairs, and building the grouped
    namespace listing used by the onboarding UI.
    """

    def arxiv_category(self, namespace_key: str) -> str | None:
        """Return the arXiv category string for a given namespace key.

        Args:
            namespace_key: An application namespace key (e.g. ``"cs.AI"`` or
                the legacy alias ``"cs.ML"``).

        Returns:
            The corresponding arXiv category string (e.g. ``"cs.AI"``), or
            ``None`` if the key is not registered in ``NAMESPACE_TO_ARXIV``.
        """
        return NAMESPACE_TO_ARXIV.get(namespace_key)

    def resolve(self, subject: str, topic: str) -> str | None:
        """Build a namespace key from a subject prefix and topic, if registered.

        Constructs a candidate key by taking the first four lowercase characters
        of ``subject`` and appending ``.{topic}``, then checks whether that key
        exists in ``NAMESPACE_TO_ARXIV``.

        Args:
            subject: The subject area (e.g. ``"cs"`` or ``"math"``). Only the
                first four characters (lowercased) are used.
            topic: The topic suffix (e.g. ``"AI"`` or ``"LG"``).

        Returns:
            The constructed namespace key if it is registered, otherwise
            ``None``.
        """
        key = f"{subject.lower()[:4]}.{topic}"
        return key if key in NAMESPACE_TO_ARXIV else None

    def subject_topics(self) -> dict[str, list[dict]]:
        """Return {subject_label: [{key, label}, ...]} for all canonical namespaces."""
        groups: dict[str, list[dict]] = {}
        seen: set[str] = set()
        for key, arxiv_cat in NAMESPACE_TO_ARXIV.items():
            if key != arxiv_cat:
                continue  # skip legacy aliases
            prefix = _group_prefix(key)
            subject = _SUBJECT_GROUPS.get(prefix, "Other")
            if subject not in groups:
                groups[subject] = []
            label = _NS_LABELS.get(key, key)
            groups[subject].append({"key": key, "label": label})
            seen.add(key)
        # Sort entries within each group
        for entries in groups.values():
            entries.sort(key=lambda e: e["key"])
        return groups
