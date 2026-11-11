#!/usr/bin/env python3
"""Audit the recovered validation CSV without running numerical solvers.

Read-only with respect to the evidence. Outputs JSON to stdout. Thresholds
come from the archived validator snapshot, whose hash is recorded below;
the original suite summary did not record the validation driver's hash.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "results/Vanda_eigen_1526107/validation/comparisons.csv"
    summary = json.loads((root / "results/Vanda_eigen_1526107/validation/summary.json").read_text())
    validator = root / "scripts/validate_eigen_pair.py"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == summary["comparisons"] == 48
    for key in ("site", "hostname", "job_id", "measurement_id"):
        assert {row[key] for row in rows} == {str(summary[key])}, key
    assert summary["status"] == "PASS" and not summary["serial_only"]
    sources_match = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in summary["source_sha256"].items()
    )
    assert sources_match
    specs = {"zero": 3, "single": 1, "tiny": 2, "path": 12,
             "cycle": 10, "self_isolated": 5}
    assert set(row["graph"] for row in rows) == set(specs)
    endpoints = []
    first_deltas = []
    for graph, n in specs.items():
        group = [row for row in rows if row["graph"] == graph]
        assert len(group) == 8
        expected = Counter(
            (algorithm, parallel, budget)
            for algorithm in ("power", "lanczos")
            for parallel in ("False", "True")
            for budget in (1, 1000 if algorithm == "power" else n)
        )
        actual = Counter((r["algorithm"], r["parallel"], int(r["budget"])) for r in group)
        assert actual == expected, graph
        for row in group:
            assert all(math.isfinite(float(row[key])) for key in ("lambda_ritz", "dense_residual"))
            assert float(row["dense_residual"]) >= 0
        # This is the validator's acceptance scope. A general one-step result
        # is a consistency check and is not required to be converged.
        selected = [row for row in group if int(row["budget"]) > 1 or n == 1]
        eigenvalue = 0.5 if graph in ("zero", "single") else 1.0
        eig_error = max(abs(float(r["lambda_ritz"]) - eigenvalue) for r in selected)
        residual = max(float(r["dense_residual"]) for r in selected)
        passed = eig_error < 1e-8 and residual < 1e-8
        assert passed, graph
        endpoints.append(dict(graph=graph, N=n, total_records=len(group),
                              endpoint_records=len(selected), expected_eigenvalue=eigenvalue,
                              max_eigenvalue_error=eig_error, max_dense_residual=residual,
                              acceptance_threshold=1e-8, recomputed_result="PASS"))
        power = next(r for r in group if r["algorithm"] == "power" and
                     r["parallel"] == "False" and int(r["budget"]) == 1)
        lanczos = next(r for r in group if r["algorithm"] == "lanczos" and
                       r["parallel"] == "False" and int(r["budget"]) == 1)
        dl = abs(float(power["lambda_ritz"]) - float(lanczos["lambda_ritz"]))
        dr = abs(float(power["dense_residual"]) - float(lanczos["dense_residual"]))
        assert dl < 1e-12 and dr < 1e-12
        first_deltas.append(dict(graph=graph, eigenvalue_difference=dl,
                                 dense_residual_difference=dr, threshold=1e-12,
                                 recomputed_result="PASS"))
    result = dict(
        data_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        source="results/Vanda_eigen_1526107/validation/comparisons.csv", records=len(rows),
        identity={key: summary[key] for key in ("site", "hostname", "job_id", "measurement_id")},
        original_suite_result=summary["status"], source_hashes_match_summary=sources_match,
        threshold_source="scripts/validate_eigen_pair.py",
        archived_validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest(),
        threshold_provenance="Re-evaluated with archived validator criteria. The original summary does not hash the validation driver.",
        endpoint_selection="budget > 1 or N == 1; maxima over both algorithms and serial/MPI rows",
        endpoints=endpoints, serial_one_step_comparisons=first_deltas,
        checks_not_recomputed="Dense-vs-solver-reported residual difference and vector normalization error: the CSV does not contain solver-reported residuals or vectors. These checks retain original suite PASS evidence only.",
        solver_execution=False,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
