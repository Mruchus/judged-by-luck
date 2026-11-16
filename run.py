"""Run one model on one domain and variant, and save the answers to results/.

Example: python run.py --model claude-code:claude-haiku-5-5 --domain medical --variant main
"""
import argparse, json, os, random, re, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prompts import CONDITIONS, DOMAINS, build_prompt

VARIANTS = {"main": dict(warn=False, odds=True),
            "warn": dict(warn=True, odds=True),
            "no_odds": dict(warn=False, odds=False)}


def parse_json(text):
    """Return the first JSON object in a model reply, or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def mock_call(system, prompt):
    """Fake outcome-biased model, used to test the pipeline without API calls."""
    q = 7.0 + random.gauss(0, 0.7)
    if "OUTCOME:\n" in prompt and "Withheld" not in prompt:
        out = prompt.split("OUTCOME:\n")[1].split("\n")[0].lower()
        q += -2.5 if any(w in out for w in ["lost", "attack", "bleed", "sepsis", "fracture", "burst", "broke",
                                           "failed", "fired", "plan", "let go", "left", "churn", "readmit",
                                           "outage", "fraud", "missed", "found on", "dismiss", "error", "struggl",
                                           "underperform", "moved", "collapsed", "admitted", "resign"]) else 1.0
    return json.dumps({"decision_quality": max(1, min(10, round(q))), "confidence_reasonable": 70,
                       "negligence": 2, "should_have_foreseen_risk": q < 6,
                       "main_risk_factor": "irreducible_uncertainty", "recommended_action": "none",
                       "rationale": "mock", "outcome_influence": 1, "outcome_should_matter": False})


def make_client(args):
    """Return a function complete(system, prompt) -> reply text for the chosen model."""
    if args.model.startswith("claude-code:"):
        # Claude via the Claude Code CLI in non-interactive mode, with our system prompt, no tools and no user config.
        model = args.model.split(":", 1)[1]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use the CLI's own login
        cwd = tempfile.mkdtemp()

        def complete(system, prompt):
            p = subprocess.run(["claude", "-p", "--safe-mode", "--model", model, "--system-prompt", system,
                                "--tools", "", "--strict-mcp-config", "--no-session-persistence",
                                "--output-format", "json", prompt],
                               capture_output=True, text=True, timeout=600, env=env, cwd=cwd)
            try:
                d = json.loads(p.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(f"claude -p exit {p.returncode}: {(p.stderr or p.stdout)[:300]}")
            if d.get("is_error"):
                raise RuntimeError(f"claude -p error: {str(d.get('result'))[:300]}")
            return d.get("result", "")
        return complete

    if args.model.startswith("claude-"):
        # Claude via the Anthropic API (needs ANTHROPIC_API_KEY). Default sampling, since current models reject others.
        import anthropic
        client = anthropic.Anthropic(timeout=300, max_retries=4)

        def complete(system, prompt):
            r = client.messages.create(model=args.model, max_tokens=4000, system=system,
                                       messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if b.type == "text")
        return complete

    # Any OpenAI-compatible server, e.g. Ollama, vLLM or OpenRouter.
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=600)
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}

    def complete(system, prompt):
        r = client.chat.completions.create(
            model=args.model, temperature=args.temperature, max_tokens=800, **extra,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    return complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock",
                    help="'claude-code:<id>' for the Claude Code CLI, 'claude-<id>' for the Anthropic API, "
                         "anything else for an OpenAI-compatible server")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible server URL, e.g. http://localhost:11434/v1")
    ap.add_argument("--domain", choices=list(DOMAINS), default="hiring")
    ap.add_argument("--variant", choices=list(VARIANTS), default="main")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7, help="OpenAI-compatible servers only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reasoning-effort", default=None, help="OpenAI-compatible servers only, e.g. 'none'")
    ap.add_argument("--mock", action="store_true", help="use the fake model")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    ap.add_argument("--check", action="store_true", help="make one short call and exit")
    args = ap.parse_args()
    if args.mock:
        args.model = "mock"
    complete = None if args.mock else make_client(args)

    if args.check:
        print("ok:", args.model, "->", (complete("Reply with one word.", "Say ok") or "").strip()[:40])
        return

    cases = json.load(open(f"data/{args.domain}.json"))
    jobs = [(c, cond, s) for c in cases for cond in CONDITIONS for s in range(args.samples)]

    Path("results").mkdir(exist_ok=True)
    safe_model = re.sub(r"[/:]", "__", args.model)  # keep file names valid on Windows
    out = Path("results") / f"{args.domain}__{args.variant}__{safe_model}.jsonl"
    if out.exists() and not args.force:
        n_ok = sum(1 for l in open(out) if json.loads(l)["parsed"] is not None)
        if n_ok >= 0.9 * len(jobs):
            print(f"skip {out} (already done)")
            return

    def call(system, prompt):
        """Call the model with retries; ask again up to 2 times if the reply has no JSON."""
        if args.mock:
            return mock_call(system, prompt)
        err = None
        for attempt in range(6):
            try:
                text = complete(system, prompt)
                if parse_json(text) is not None or attempt >= 2:
                    return text
            except Exception as e:
                err = e
                time.sleep(min(60, 2 ** attempt))
        return f"ERROR: {err}"

    system = DOMAINS[args.domain]["system"]
    v = VARIANTS[args.variant]

    def work(job):
        case, cond, s = job
        raw = call(system, build_prompt(case, cond, args.domain, **v))
        return {"model": args.model, "domain": args.domain, "variant": args.variant,
                "case": case["id"], "prior_quality": case["prior_quality"],
                "condition": cond, "sample": s, "raw": raw, "parsed": parse_json(raw)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(work, jobs))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bad = sum(r["parsed"] is None for r in rows)
    print(f"wrote {len(rows)} rows to {out} ({bad} unparseable)")


if __name__ == "__main__":
    main()
