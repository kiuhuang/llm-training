#!/usr/bin/env python
"""Compare base vs fine-tuned model outputs and write a Markdown report.

Reads the two prediction JSONLs produced by run_inference.py (joined on "id"),
computes per-item and aggregate metrics, and writes:

  outputs/eval/comparison_report.md   human-readable report (tables + examples)
  outputs/eval/summary.json           machine-readable metrics
  outputs/eval/per_item.csv           per-prompt metrics for further analysis

Metrics:
  * ROUGE-L F1 vs the held-out reference answer (lexical overlap)
  * exact-match rate (normalized text)
  * paired win/tie/loss for tuned vs base, with bootstrap 95% CI on mean delta
  * degenerate-output rate (empty, or a line repeated 4+ times)
  * length statistics + truncation rate

  python scripts/compare_results.py \
      --base outputs/eval/base.jsonl --tuned outputs/eval/tuned.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def is_degenerate(s: str) -> bool:
    if len(s.strip()) == 0:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines and Counter(lines).most_common(1)[0][1] >= 4:
        return True
    if len(set(s.split())) * 8 < len(s.split()) and len(s.split()) > 40:
        return True  # extreme token repetition
    return False


def bootstrap_ci(deltas: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    if n < 2:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def truncate(s: str, n: int = 400) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + " …"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base model predictions JSONL")
    p.add_argument("--tuned", required=True, help="fine-tuned model predictions JSONL")
    p.add_argument("--base-label", default="base")
    p.add_argument("--tuned-label", default="tuned")
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--examples", type=int, default=8, help="side-by-side examples in the report")
    args = p.parse_args()

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def load(path):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[r["id"]] = r
        return rows

    base_rows, tuned_rows = load(args.base), load(args.tuned)
    common = sorted(set(base_rows) & set(tuned_rows))
    if len(common) < len(base_rows) or len(common) < len(tuned_rows):
        print(f"[warn] id overlap: {len(common)} (base={len(base_rows)}, tuned={len(tuned_rows)})")

    per_item = []
    for rid in common:
        b, t = base_rows[rid], tuned_rows[rid]
        ref = b.get("reference", "")
        r_b = scorer.score(ref, b["prediction"])["rougeL"].fmeasure
        r_t = scorer.score(ref, t["prediction"])["rougeL"].fmeasure
        em_b = float(norm_text(b["prediction"]) == norm_text(ref))
        em_t = float(norm_text(t["prediction"]) == norm_text(ref))
        per_item.append({
            "id": rid,
            "rougeL_base": r_b, "rougeL_tuned": r_t, "rougeL_delta": r_t - r_b,
            "exact_base": em_b, "exact_tuned": em_t,
            "degenerate_base": is_degenerate(b["prediction"]),
            "degenerate_tuned": is_degenerate(t["prediction"]),
            "len_base": len(b["prediction"].split()), "len_tuned": len(t["prediction"].split()),
            "len_ref": len(ref.split()),
            "truncated_base": b.get("truncated", False), "truncated_tuned": t.get("truncated", False),
        })

    n = len(per_item)
    if n == 0:
        raise SystemExit("no overlapping ids between the two prediction files")

    def agg(key):
        vals = [it[key] for it in per_item]
        return sum(vals) / n

    deltas = [it["rougeL_delta"] for it in per_item]
    mean_d = sum(deltas) / n
    ci_lo, ci_hi = bootstrap_ci(deltas)
    wins = sum(d > 1e-6 for d in deltas)
    ties = sum(abs(d) <= 1e-6 for d in deltas)
    losses = sum(d < -1e-6 for d in deltas)

    summary = {
        "base_file": args.base, "tuned_file": args.tuned, "n_pairs": n,
        "rougeL": {args.base_label: agg("rougeL_base"), args.tuned_label: agg("rougeL_tuned"),
                   "delta_mean": mean_d, "delta_ci95": [ci_lo, ci_hi]},
        "exact_match": {args.base_label: agg("exact_base"), args.tuned_label: agg("exact_tuned")},
        "win_tie_loss": {args.base_label: losses, "tie": ties, args.tuned_label: wins},
        "degenerate_rate": {args.base_label: agg("degenerate_base"),
                            args.tuned_label: agg("degenerate_tuned")},
        "truncation_rate": {args.base_label: agg("truncated_base"),
                            args.tuned_label: agg("truncated_tuned")},
        "mean_len_words": {args.base_label: agg("len_base"), args.tuned_label: agg("len_tuned"),
                           "reference": agg("len_ref")},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "per_item.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_item[0].keys()))
        w.writeheader()
        w.writerows(per_item)

    # ---------------- Markdown report ----------------
    b_lab, t_lab = args.base_label, args.tuned_label
    lines = [
        "# Base vs Fine-tuned — Evaluation Report",
        "",
        f"- base file : `{args.base}`",
        f"- tuned file: `{args.tuned}`",
        f"- evaluated pairs: **{n}** (held-out test prompts, greedy decoding)",
        "",
        "## Headline metrics",
        "",
        f"| metric | {b_lab} | {t_lab} | delta |",
        "|---|---|---|---|",
        f"| ROUGE-L F1 vs reference | {summary['rougeL'][b_lab]:.4f} | "
        f"{summary['rougeL'][t_lab]:.4f} | **{mean_d:+.4f}** "
        f"(95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]) |",
        f"| exact match | {summary['exact_match'][b_lab]:.3f} | {summary['exact_match'][t_lab]:.3f} | "
        f"{summary['exact_match'][t_lab] - summary['exact_match'][b_lab]:+.3f} |",
        f"| degenerate outputs | {summary['degenerate_rate'][b_lab]:.3f} | "
        f"{summary['degenerate_rate'][t_lab]:.3f} | — |",
        f"| truncation rate | {summary['truncation_rate'][b_lab]:.3f} | "
        f"{summary['truncation_rate'][t_lab]:.3f} | — |",
        f"| mean length (words) | {summary['mean_len_words'][b_lab]:.0f} | "
        f"{summary['mean_len_words'][t_lab]:.0f} | (reference: "
        f"{summary['mean_len_words']['reference']:.0f}) |",
        "",
        f"**Paired result:** {t_lab} wins on **{wins}** / ties **{ties}** / loses **{losses}** "
        f"of {n} prompts (ROUGE-L).",
        "",
        "> Interpretation: ROUGE-L measures lexical overlap with one reference answer — "
        "a proxy, not a verdict. Read it together with the samples below and "
        "(optionally) an LLM judge (docs/04_evaluation.md).",
        "",
        "## Side-by-side samples",
        "",
    ]
    # show biggest wins first, then losses — most informative examples
    ranked = sorted(per_item, key=lambda it: it["rougeL_delta"], reverse=True)
    picks: list[int] = []
    half = max(1, args.examples // 2)
    picks += [it["id"] for it in ranked[:half]]
    picks += [it["id"] for it in ranked[-half:]]
    for rid in picks:
        b, t = base_rows[rid], tuned_rows[rid]
        it = next(i for i in per_item if i["id"] == rid)
        q = next((m["content"] for m in b["messages"] if m["role"] == "user"), "?")
        lines += [
            f"### id {rid}  (Δ ROUGE-L {it['rougeL_delta']:+.3f})",
            "",
            f"**Prompt:** {truncate(q, 300)}",
            "",
            f"**Reference:** {truncate(b.get('reference', ''), 300)}",
            "",
            f"**{b_lab}:** {truncate(b['prediction'], 400)}",
            "",
            f"**{t_lab}:** {truncate(t['prediction'], 400)}",
            "",
        ]

    report = os.path.join(args.out_dir, "comparison_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[write] {report}")
    print(f"[write] {os.path.join(args.out_dir, 'summary.json')}")
    print(f"\nROUGE-L: {b_lab}={summary['rougeL'][b_lab]:.4f}  "
          f"{t_lab}={summary['rougeL'][t_lab]:.4f}  delta={mean_d:+.4f} "
          f"(win/tie/loss = {wins}/{ties}/{losses})")


if __name__ == "__main__":
    main()
