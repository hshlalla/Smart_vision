## Report Support Artifacts

Created on: 2026-03-10

Purpose:
- Preserve reproducible evidence for the March 10, 2026 validation pass.
- Provide raw test outputs and machine-readable JUnit XML for report writing.
- Record the code changes that were validated in this pass.

Files:
- `api_pytest_output.txt`: raw console output from `apps/api` pytest run.
- `api_pytest.junit.xml`: JUnit XML from `apps/api` pytest run.
- `model_pytest_output.txt`: raw console output from `packages/model` pytest run.
- `model_pytest.junit.xml`: JUnit XML from `packages/model` pytest run.
- `test_environment.txt`: Python and package versions used for the recorded runs.
- `change_summary.md`: summary of the product-safety and testability changes validated here.

Commands used:
- `cd /Users/mac/project/Smart_vision/apps/api && /tmp/sv-test-min/bin/python -m pytest -q -p no:cacheprovider --junitxml=artifacts/report_support_2026-03-10/api_pytest.junit.xml`
- `cd /Users/mac/project/Smart_vision/packages/model && /tmp/sv-test-min/bin/python -m pytest -q -p no:cacheprovider --junitxml=../../apps/api/artifacts/report_support_2026-03-10/model_pytest.junit.xml`

Recorded results:
- API tests: `12 passed, 1 warning in 5.37s`
- Model tests: `4 passed in 0.09s`

Notes:
- The test environment was created in `/tmp/sv-test-min`.
- Model package collection originally hit eager-import issues around heavyweight runtime dependencies. Lazy imports were applied before the final recorded passing run.
