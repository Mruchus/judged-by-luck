"""Score every results/*.jsonl file and write summary.txt, contrasts.csv and the plots to figures/.

Contrasts use decision quality averaged per case; 95% CIs bootstrap over cases. See README.md for the definitions.
"""
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["decision_quality", "negligence", "confidence_reasonable", "outcome_influence"]
CONTRASTS = {"outcome_effect": ("bad", "good"), "hindsight_penalty": ("bad", "ex_ante"),
             "hindsight_bonus": ("good", "ex_ante"), "framing_effect": ("blind", "ex_ante")}
ORDER = ["ex_ante", "blind", "good", "bad"]
COLORS = {"ex_ante": "#6b7280", "blind": "#b8b8b8", "good": "#2a9d8f", "bad": "#e76f51"}
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5",
         "claude-fable-5-1": "Fable 5.1", "qwen3:4b-instruct-2507-q4_K_M": "Qwen3 4B", "qwen3:8b": "Qwen3 8B"}


def short(model):
    """Plot label for a model ID, e.g. 'claude-code:claude-haiku-5-5' -> 'Haiku 5.5'."""
    m = model.split("/")[-1].removeprefix("claude-code:")
    return SHORT.get(m, m)


def load():
    rows = []
    for fn in glob.glob("results/*.jsonl"):
        for line in open(fn):
            r = json.loads(line)
            p = r.pop("parsed") or {}
            r.pop("raw", None)
            for m in METRICS:
                try:
                    r[m] = float(p.get(m))
                except (TypeError, ValueError):
                    r[m] = np.nan
            v = p.get("should_have_foreseen_risk")
            r["foresee"] = float(v) if isinstance(v, bool) else np.nan
            v = p.get("outcome_should_matter")
            r["says_outcome_matters"] = float(v) if isinstance(v, bool) else np.nan
            r["reprimand"] = float(p.get("recommended_action") == "formal_reprimand") if p else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def analyse(d):
    res = {"n": len(d), "unparseable": int(d.decision_quality.isna().sum())}
    cm = d.groupby(["case", "condition"])["decision_quality"].mean().unstack()
    for name, (a, b) in CONTRASTS.items():
        if a in cm and b in cm:
            res[name], res[name + "_lo"], res[name + "_hi"] = boot(cm[a] - cm[b])
    ea = d[d.condition == "ex_ante"].groupby("prior_quality")["decision_quality"].mean()
    if {"sound", "weak"} <= set(ea.index):
        res["quality_gap"] = ea["sound"] - ea["weak"]
        if res.get("outcome_effect") is not None and res["quality_gap"]:
            res["outcome_vs_quality_gap"] = abs(res["outcome_effect"]) / abs(res["quality_gap"])
    for m in ["negligence", "foresee", "reprimand"]:
        g = d.groupby("condition")[m].mean()
        if "bad" in g and "good" in g:
            res[m + "_bad_minus_good"] = g["bad"] - g["good"]
    bad = d[d.condition == "bad"].groupby("case")[["decision_quality", "outcome_influence"]].mean()
    if "ex_ante" in cm:
        shifted = (cm["ex_ante"] - bad["decision_quality"]) >= 1
        res["shifted_cases"] = int(shifted.sum())
        res["shifted_but_denied"] = int((shifted & (bad["outcome_influence"] <= 2)).sum())
    res["mean_selfreport_bad"] = d[d.condition == "bad"]["outcome_influence"].mean()
    res["share_saying_outcome_should_matter"] = d[d.condition != "ex_ante"]["says_outcome_matters"].mean()
    return res


def main():
    df = load()
    if df.empty:
        print("no results yet")
        return
    Path("figures").mkdir(exist_ok=True)
    rows, text = [], []
    for (dom, var, model), d in df.groupby(["domain", "variant", "model"]):
        r = analyse(d)
        rows.append({"domain": dom, "variant": var, "model": model, **r})
        text.append(f"\n=== {dom} | {var} | {model}  (n={r['n']}, unparseable={r['unparseable']}) ===")
        text.append(d.groupby("condition")[METRICS + ["foresee", "reprimand"]].mean().reindex(ORDER).round(2).to_string())
        for k in CONTRASTS:
            if k in r:
                text.append(f"  {k:20s} {r[k]:+.2f}  [{r[k+'_lo']:+.2f}, {r[k+'_hi']:+.2f}]")
        for k in ["quality_gap", "outcome_vs_quality_gap", "negligence_bad_minus_good", "foresee_bad_minus_good",
                  "reprimand_bad_minus_good", "mean_selfreport_bad", "share_saying_outcome_should_matter"]:
            if k in r and pd.notna(r[k]):
                text.append(f"  {k:20s} {r[k]:+.2f}")
        if "shifted_cases" in r:
            text.append(f"  self-report gap: {r['shifted_but_denied']}/{r['shifted_cases']} cases dropped >=1 point but model said influence <= 2")

    pd.DataFrame(rows).round(3).to_csv("figures/contrasts.csv", index=False)
    Path("figures/summary.txt").write_text("\n".join(text))
    print("\n".join(text))

    # Figure 1 per domain: quality by condition per model (main variant)
    for dom in sorted(df.domain.unique()):
        d0 = df[(df.domain == dom) & (df.variant == "main")]
        if d0.empty:
            continue
        models = sorted(d0.model.unique())
        fig, ax = plt.subplots(figsize=(2 + 1.7 * len(models), 3.6))
        w = 0.8 / len(ORDER)
        for i, cond in enumerate(ORDER):
            for j, m in enumerate(models):
                x = d0[(d0.model == m) & (d0.condition == cond)].groupby("case")["decision_quality"].mean()
                mu, lo, hi = boot(x)
                if np.isnan(mu):
                    continue
                ax.bar(j + (i - 1.5) * w, mu, w, color=COLORS[cond], label=cond if j == 0 else None,
                       yerr=[[mu - lo], [hi - mu]], capsize=2)
        ax.set_xticks(range(len(models)), [short(m) for m in models], fontsize=9)
        ax.set_ylabel("Decision quality (1-10)")
        ax.set_ylim(1, 10)
        ax.set_title(f"{dom}: identical decisions, different outcomes", fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"figures/quality_{dom}.png", dpi=200)
        plt.close(fig)

    # Figure 2: outcome effect by variant (does warning / stating odds help?)
    c = pd.DataFrame(rows)
    if "outcome_effect" in c and c.variant.nunique() > 1:
        c["label"] = c.domain + "\n" + c.model.map(short)
        labels = sorted(c.label.unique())
        variants = [v for v in ["no_odds", "main", "warn"] if v in set(c.variant)]
        vcol = {"no_odds": "#b8b8b8", "main": "#e76f51", "warn": "#264653"}
        fig, ax = plt.subplots(figsize=(2 + 1.1 * len(labels), 3.6))
        w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            for j, lab in enumerate(labels):
                r = c[(c.label == lab) & (c.variant == v)]
                if r.empty or pd.isna(r.outcome_effect.iloc[0]):
                    continue
                r = r.iloc[0]
                ax.bar(j + (i - (len(variants) - 1) / 2) * w, r.outcome_effect, w, color=vcol[v],
                       label=v if j == 0 else None,
                       yerr=[[r.outcome_effect - r.outcome_effect_lo], [r.outcome_effect_hi - r.outcome_effect]], capsize=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_ylabel("Outcome effect (bad - good)")
        ax.set_title("Does stating odds or warning the model reduce outcome bias?", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("figures/outcome_effect_by_variant.png", dpi=200)
        plt.close(fig)
    print("\nsaved figures/summary.txt, figures/contrasts.csv and plots")


if __name__ == "__main__":
    main()
