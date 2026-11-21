# Contributing

For a reproducibility issue, include the commit, OS, Python and compiler versions, exact command, configuration, expected result, and relevant log. Avoid attaching access tokens, model weights, or copyrighted corpus text.

Install the Python environment from the README, run `python -m pytest -q`, and run the checks relevant to the changed layer. Simulator changes require `python -m tools.build` and `python -m tools.reproduce` for every affected suite (each run is audited against its raw trace and compared with the reference). RTL changes require `python -m tools.verify_rtl` or the calibration checks, plus regenerated timing calibration or area components when they change. Changes to measured tables must identify the raw measurement source and its hashes.

Keep reference results immutable. Write new runs to `outputs/`, label analytical estimates explicitly, and explain any changes to precision, tie-breaking, workload, refresh policy, or area scope. Include the test command and result in pull requests.

`requirements-lock.txt` freezes the resolved Python 3.11 environment; `requirements-reproduce.txt` records its direct dependency choices. Update and verify both intentionally when changing dependencies.
