import re

# Example: keywords that, if present, force High risk (add as needed)
FORCED_HIGH_KEYWORDS = {
    "explosion", "crush hazard", "fatal", "death", "amputation", "toxic gas",
    "lethal voltage", "electrocution", "fire/explosion", "uncontrolled motion"
}

# Example: Department-mapping keywords
DEPARTMENT_KEYWORDS = {
    "controls": {"plc", "safety relay", "safety circuit", "interlock", "e-stop", "robot"},
    "mechanical": {"gear", "shaft", "pneumatic", "hydraulic", "mechanism", "bearing"},
    "safety": {"guard", "light curtain", "barrier", "lockout", "pinch point", "safety"},
    "electrical": {"voltage", "current", "arc flash", "fuse", "wiring", "breaker"},
}

# Regex patterns for critical situations (expand as needed)
REGEX_PATTERNS = [
    (re.compile(r"\b(5000\s*psi|high\s*pressure|hydraulic leak)\b", re.IGNORECASE), "mechanical", "High"),
    (re.compile(r"\b(exposed wires?|uninsulated)\b", re.IGNORECASE), "electrical", "High"),
]

# Example: phrase sets for forced assignment (expandable)
FORCED_ASSIGNMENTS = [
    # (phrase_set, forced_risk_level, forced_department)
    ({"arc flash", "live busbar"}, "High", "electrical"),
    ({"unguarded robot", "no light curtain"}, "High", "controls"),
]

def apply_rules(text):
    """
    Checks the text for known keywords, regex, and phrase triggers.
    Returns:
        dict with:
            - triggered: bool (did any rule fire)
            - reasons: list of str (what matched)
            - forced_label: dict with 'Risk Level' and/or 'Review Department' (if any)
    """
    text_lower = text.lower()
    triggered = False
    reasons = []
    forced_label = {}

    # 1. Forced high risk by keywords
    for kw in FORCED_HIGH_KEYWORDS:
        if kw in text_lower:
            triggered = True
            reasons.append(f'Keyword "{kw}" triggers High risk.')
            forced_label['Risk Level'] = "high"

    # 2. Department assignment by keywords
    for dept, kwset in DEPARTMENT_KEYWORDS.items():
        for kw in kwset:
            if kw in text_lower:
                triggered = True
                reasons.append(f'Keyword "{kw}" triggers department "{dept}".')
                forced_label.setdefault("Review Department", dept.lower().strip())

    # 3. Regex patterns
    for pattern, dept, risk in REGEX_PATTERNS:
        if pattern.search(text):
            triggered = True
            reasons.append(f'Regex pattern "{pattern.pattern}" triggers {risk} risk, department {dept}.')
            forced_label['Risk Level'] = risk.lower().strip()
            forced_label['Review Department'] = dept.lower().strip()

    # 4. Forced assignments via phrase set
    for phrases, risk, dept in FORCED_ASSIGNMENTS:
        if any(phrase in text_lower for phrase in phrases):
            triggered = True
            reasons.append(f'Phrases {phrases} force {risk} risk, department {dept}.')
            forced_label['Risk Level'] = risk.lower().strip()
            forced_label['Review Department'] = dept.lower().strip()

    # 5. Heuristic rules (example: too short = needs review)
    if len(text_lower.split()) < 3:
        triggered = True
        reasons.append("Description is very short—force Manual Review.")
        forced_label['Risk Level'] = "manual review"
        forced_label['Review Department'] = "unclassified"

    

    # Ensure forced labels are always lowercase/stripped if set
    for k in list(forced_label.keys()):
        if isinstance(forced_label[k], str):
            forced_label[k] = forced_label[k].lower().strip()

    return {
        "triggered": triggered,
        "reasons": reasons,
        "forced_label": forced_label
    }

# Utility to run on a list of texts for batch processing:
def batch_apply_rules(texts):
    """
    Apply rules to a list/Series of texts.
    Returns list of rule results.
    """
    return [apply_rules(t) for t in texts]
