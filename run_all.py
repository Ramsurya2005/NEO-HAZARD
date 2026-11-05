"""Run the completed NEO analysis workflow in reproducible order."""

from src.assumptions import run_assumption_checks
from src.eda import run_eda
from src.explainability import run_explainability
from src.inference import run_inference
from src.leakage_demo import run_leakage_demo
from src.modeling import run_modeling
from src.reproducibility import run_reproducibility_checks


def main() -> None:
    """Run Steps 2–7 and the Step 10 reproducibility checks."""
    print("Step 2: EDA")
    run_eda()
    print("Step 3: assumption checks")
    run_assumption_checks()
    print("Step 4: inferential statistics")
    run_inference()
    print("Step 5: predictive modelling")
    run_modeling()
    print("Step 6: leakage demonstration")
    run_leakage_demo()
    print("Step 7: explainability")
    run_explainability()
    print("Step 10: reproducibility checks")
    run_reproducibility_checks()
    print("All requested non-optional steps completed.")


if __name__ == "__main__":
    main()
