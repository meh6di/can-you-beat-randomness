"""Run the whole pipeline in order.

  python run_all.py                  # everything
  python run_all.py --skip-fetch     # reuse data_raw/
  python run_all.py --from walkforward
  python run_all.py --only market_bias
"""
import subprocess
import sys

STEPS = [
    ("fetch", "fetch_data.py"), ("clean", "clean_data.py"), ("features", "engineer_features.py"),
    ("walkforward", "walkforward.py"), ("stats", "stats_tests.py"), ("prob_eval", "prob_eval.py"),
    ("clv", "clv_analysis.py"), ("select", "strategy_select.py"), ("market_bias", "market_bias.py"),
    ("risk", "risk_analysis.py"), ("tables", "make_report_tables.py"),
]


def main():
    names = [s for s, _ in STEPS]
    args = sys.argv[1:]
    todo = STEPS
    if "--only" in args:
        todo = [s for s in STEPS if s[0] == args[args.index("--only") + 1]]
    elif "--from" in args:
        todo = STEPS[names.index(args[args.index("--from") + 1]):]
    if "--skip-fetch" in args:
        todo = [s for s in todo if s[0] != "fetch"]
    if not todo:
        raise SystemExit(f"Unknown step. Choose from: {', '.join(names)}")
    for name, script in todo:
        print(f"\n{'=' * 70}\n>> {name}: {script}\n{'=' * 70}", flush=True)
        if subprocess.run([sys.executable, script]).returncode != 0:
            raise SystemExit(f"Step '{name}' failed.")
    print("\nAll done. See results/ (summary.md has every table).")


if __name__ == "__main__":
    main()
