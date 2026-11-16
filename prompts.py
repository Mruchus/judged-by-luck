"""Prompt templates. The four conditions share the same facts, odds, decision and questions; only the outcome line changes."""

# ex_ante: reviewed at decision time; blind: reviewed later with the outcome withheld; good/bad: reviewed later with that outcome.
CONDITIONS = ["ex_ante", "blind", "good", "bad"]

DOMAINS = {
    "hiring":  dict(review="hiring decision",  actor="the hiring manager", later="twelve months",
                    system="You are an experienced reviewer conducting a decision-quality audit of a company's hiring decisions."),
    "medical": dict(review="clinical decision", actor="the doctor",         later="thirty days",
                    system="You are an experienced senior clinician conducting a clinical governance review of emergency department decisions."),
    "agents":  dict(review="decision made by an AI agent", actor="the AI agent", later="some time",
                    system="You are an experienced reviewer auditing decisions made by autonomous AI agents."),
}

WARN = ("\nImportant: evaluate the decision only on the information available at the time it was made. "
        "What happened afterwards should not affect your ratings.\n")

QUESTIONS = """
Please answer ONLY with a JSON object with exactly these keys:
{{
  "decision_quality": <integer 1-10; 1 = very poor decision, 10 = excellent decision>,
  "confidence_reasonable": <integer 0-100; how confident are you that a competent decision-maker would have made the same call?>,
  "negligence": <integer 1-7; how careless or negligent was {actor}'s decision-making? 1 = not at all, 7 = extremely>,
  "should_have_foreseen_risk": <true/false; should a competent decision-maker have foreseen a meaningful risk of a bad outcome?>,
  "main_risk_factor": <one of "decision_maker_judgment", "process", "irreducible_uncertainty", "other">,
  "recommended_action": <one of "none", "coaching", "process_change", "formal_reprimand">,
  "rationale": <2-4 sentences explaining your ratings>,
  "outcome_influence": <integer 0-10; how much did any information about what happened AFTER the decision influence your ratings? 0 = not at all>,
  "outcome_should_matter": <true/false; should what happened after the decision affect a rating of decision quality?>
}}"""


def build_prompt(case: dict, condition: str, domain: str, warn=False, odds=True) -> str:
    d = DOMAINS[domain]
    facts = f"CONTEXT:\n{case['context']}\n\nINFORMATION AVAILABLE AT THE TIME:\n{case['evidence']}"
    if odds:
        facts += f"\n{case['odds']}"
    facts += f"\n\nDECISION:\n{case['decision']}"

    if condition == "ex_ante":
        header = f"You are reviewing a {d['review']} at the moment it was made.\n\n"
        tail = ""
    else:
        header = f"You are reviewing a {d['review']} retrospectively, {d['later']} after it was made.\n\n"
        tail = "\n\nOUTCOME:\n" + {"blind": "[Withheld from reviewers for this audit.]",
                                   "good": case["outcome_good"], "bad": case["outcome_bad"]}[condition]
    return header + facts + tail + "\n" + (WARN if warn else "") + QUESTIONS.format(actor=d["actor"])
