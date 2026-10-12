"""Item 9/10 - Turn results/*.csv into report-ready tables so no number is ever hand-copied.

Writes  results/summary.md           (all tables, all columns, GitHub-flavoured markdown)
        results/tex/<name>.tex       (one complete LaTeX table per CSV; \\input'ed by report.tex)
Missing CSVs are skipped. Usage: python make_report_tables.py

Each .tex file is a full `table` environment (with caption and label), so report.tex must
\\input it directly and NOT wrap it in another table/resizebox.
Wide tables are kept readable in LaTeX by (a) showing a column subset (`tex_cols`; summary.md
still has every column), (b) shortening long labels (SHORT), or (c) transposing (`transpose`).
"""
import os

import pandas as pd

import config as C

R = C.RESULTS_DIR

FULL = "full holdout"
CLOSE = "898 holdout matches with Pinnacle closing odds"

# name -> dict(caption, pct=[percent columns], tex_cols=[subset for LaTeX], transpose=bool)
TABLES = {
    "stats_overall": dict(
        caption=f"ROI with bootstrap CI and efficient-market null ({FULL}, 1 bet/match, edge rule).",
        pct=["roi", "ci_lo", "ci_hi", "null_mean"]),
    "stats_per_league": dict(
        caption=f"Per-league ROI at the 6% threshold ({FULL}): note how wide the intervals are.",
        pct=["roi", "ci_lo", "ci_hi"]),
    "stats_paired": dict(
        caption=f"Paired bootstrap of total profit, A minus B in units ({FULL}, 6% threshold).",
        pct=[]),
    "stats_power": dict(
        caption=f"Power analysis ({FULL}, 6% threshold): what this sample can and cannot detect.",
        pct=["se_roi", "min_detectable_roi"]),
    "prob_scores": dict(
        caption="Proper scoring rules versus the market (lower is better).", pct=[]),
    "calibration": dict(
        caption="Calibration error (ECE) against its sampling-noise floor.", pct=[]),
    "clv_summary": dict(
        caption=(f"Closing line value against Pinnacle's closing price ({CLOSE}; 6% edge rule). "
                 "ROI is computed at average odds, so it is blank in the best-odds rows."),
        pct=["clv", "clv_lo", "clv_hi", "pct_clv_pos", "pct_beat_raw_close", "roi", "se_clv", "se_roi",
             "pick_skill", "pick_skill_lo", "pick_skill_hi"],
        tex_cols=["strategy", "odds_used", "bets", "clv", "clv_lo", "clv_hi", "pct_beat_raw_close", "roi",
                  "se_ratio_roi_over_clv", "pick_skill", "pick_skill_lo", "pick_skill_hi", "pick_skill_p"]),
    "select_holdout": dict(
        caption=f"Strategy chosen on dev (CLV), evaluated once on the holdout ({CLOSE}).",
        pct=["dev_clv", "hold_roi", "roi_lo", "roi_hi", "null_mean", "hold_clv", "clv_lo", "clv_hi",
             "hold_pick_skill"],
        transpose=True),
    "market_bias_buckets": dict(
        caption="Blind betting by odds bucket (favourite-longshot bias; all matches, one row per outcome).",
        pct=[c for p in ("Avg", "PS", "Max") for c in (f"roi_{p}", f"roi_{p}_lo", f"roi_{p}_hi")]
            + ["implied_raw", "fair_prob", "win_freq", "win_freq_se"],
        tex_cols=["bucket", "n", "fair_prob", "win_freq", "roi_Avg", "roi_Avg_lo", "roi_Avg_hi",
                  "roi_PS", "roi_Max"]),
    "market_bias_strategies": dict(
        caption="Simple blind strategies over all matches.", pct=["roi", "ci_lo", "ci_hi"]),
    "risk_drawdown": dict(
        caption="Flat-stake drawdown and final P&L distribution (units; logistic 6% edge strategy, full holdout).",
        pct=[]),
    "risk_ruin": dict(
        caption="Probability of ruin by bankroll size (flat 1-unit stakes; logistic 6% edge strategy, full holdout).",
        pct=[]),
    "risk_staking": dict(
        caption="Staking schemes from a 100-unit bankroll (logistic 6% edge strategy, full holdout).",
        pct=[]),
}

# LaTeX-only label shortening (summary.md keeps the full text)
SHORT = {
    "Average market odds (as in the backtests)": "Avg",
    "Best available odds (line-shopping, optimistic)": "Best",
    " (scaled, tuned)": "",
    " (expected, all matches)": " (expected)",
    " (all matches)": "",
}

# LaTeX-only header shortening (summary.md keeps the CSV column names)
HEADERS = {
    "se_ratio_roi_over_clv": "SE ratio", "pct_beat_raw_close": "beat close",
    "pick_skill": "skill", "pick_skill_lo": "skill lo", "pick_skill_hi": "skill hi", "pick_skill_p": "skill p",
    "odds_used": "odds", "clv_lo": "clv lo", "clv_hi": "clv hi",
    "roi_Avg": "ROI Avg", "roi_Avg_lo": "lo", "roi_Avg_hi": "hi", "roi_PS": "ROI PS", "roi_Max": "ROI Max",
    "fair_prob": "fair p", "win_freq": "win freq",
}


def fmt(v, pct):
    if isinstance(v, (bool,)):
        return str(v)
    if pd.isna(v):
        return "-"
    if isinstance(v, (int,)) or (hasattr(v, "dtype") and "int" in str(v.dtype)):
        return str(int(v))
    if isinstance(v, float):
        return f"{100 * v:.2f}\\%" if pct else f"{v:.3f}"
    return str(v)


def esc(s):
    for a, b in (("\\%", "%"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        s = s.replace(a, b)
    return s


def short(s):
    for a, b in SHORT.items():
        s = s.replace(a, b)
    return s


def hdr(c):
    return esc(HEADERS.get(c, c))


def cell(v, is_pct):
    return esc(short(fmt(v, is_pct))).replace("\\\\%", "\\%")


def wrap(body_lines, caption, name):
    return "\n".join([r"\begin{table}[H]\centering\small", r"\resizebox{\textwidth}{!}{%"]
                     + body_lines
                     + [r"}", rf"\caption{{{esc(caption)}}}\label{{tab:{name}}}", r"\end{table}"])


def to_latex(df, caption, name, pct_cols):
    cols = list(df.columns)
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(hdr(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(cell(r[c], c in pct_cols) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return wrap(lines, caption, name)


def to_latex_transposed(df, caption, name, pct_cols):
    """Few rows, many columns: one row per original column, one column per original row."""
    head = [esc(str(df.iloc[i][df.columns[0]])) for i in range(len(df))]
    lines = [rf"\begin{{tabular}}{{l{'r' * len(df)}}}", r"\toprule",
             esc(df.columns[0]) + " & " + " & ".join(head) + r" \\", r"\midrule"]
    for c in df.columns[1:]:
        lines.append(esc(c) + " & " + " & ".join(cell(df.iloc[i][c], c in pct_cols) for i in range(len(df))) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return wrap(lines, caption, name)


def to_markdown(df, pct_cols):
    def f(v, c):
        return fmt(v, c in pct_cols).replace("\\%", "%")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    body = "\n".join("| " + " | ".join(f(r[c], c) for c in df.columns) + " |" for _, r in df.iterrows())
    return head + "\n" + body


def clean(name, df):
    """clv_analysis.py prices profit at average odds even for the best-odds rows; blank those ROI cells."""
    if name == "clv_summary" and "odds_used" in df:
        best = df["odds_used"].astype(str).str.startswith("Best")
        for c in ("roi", "se_roi", "se_ratio_roi_over_clv"):
            if c in df:
                df.loc[best, c] = float("nan")
    return df


def main():
    os.makedirs(f"{R}/tex", exist_ok=True)
    md = ["# Results summary (auto-generated by make_report_tables.py)\n"]
    n = 0
    for name, spec in TABLES.items():
        path = f"{R}/{name}.csv"
        if not os.path.exists(path):
            continue
        df = clean(name, pd.read_csv(path))
        pct, caption = spec["pct"], spec["caption"]
        tdf = df[spec["tex_cols"]] if spec.get("tex_cols") else df
        build = to_latex_transposed if spec.get("transpose") else to_latex
        with open(f"{R}/tex/{name}.tex", "w") as f:
            f.write(build(tdf, caption, name, pct))
        md.append(f"## {caption}\n\n{to_markdown(df, pct)}\n")
        n += 1
    with open(f"{R}/summary.md", "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {n} tables to {R}/tex/ and {R}/summary.md")


if __name__ == "__main__":
    main()
