#!/usr/bin/env python3
"""Summarize supplied GPU Lanczos CSV evidence without executing solvers."""
import csv
import hashlib
import json
import math
import statistics
import runpy
from collections import Counter, defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/Vanda_gpu_lanczos_1536096"
    def read(name):
        with (folder / ("validation/comparisons.csv" if name == "comparisons.csv" else name)).open(newline="") as stream:
            return list(csv.DictReader(stream))
    runs, summary, validation = (read(n) for n in ("runs.csv", "summary.csv", "comparisons.csv"))
    assert (len(runs), len(summary), len(validation)) == (300, 60, 48)
    variants = {"cpu_serial", "cpu_mpi2_omp2", "gpu_mpi1", "gpu_mpi2_shared"}
    models = {"uniform", "hub", "community"}
    assert {r["variant"] for r in runs} == variants
    assert {r["model"] for r in runs} == models
    for rows in (runs, summary, validation):
        assert {r["site"] for r in rows} == {"Vanda"}
        assert {r["job_id"] for r in rows} == {"1536096.stdct-mgmt-02"}
    assert {r["hostname"] for r in runs} == {"GN-A40-003"}
    assert {r["precision"] for r in runs} == {"fp64"}
    assert {r["operator"] for r in runs} == {"lazy"}
    assert {r["reorth"] for r in runs} == {"full"}
    assert {r["stage"] for r in runs} == {"full"}
    assert {int(r["budget"]) for r in runs} == {5, 10, 20, 30, 50}
    assert {(r["N"], r["input_E"], r["seed"]) for r in runs} == {("100000", "800000", "20261029")}
    gpu = [r for r in runs if r["variant"].startswith("gpu")]
    assert {r["gpu_name"] for r in gpu} == {"NVIDIA A40"}
    assert len({r["gpu_uuid"] for r in gpu}) == 1
    assert {r["shared_physical_gpus"] for r in gpu} == {"1"}
    assert {r["measurement_id"] for r in runs} == {r["measurement_id"] for r in summary}
    measurement_id = runs[0]["measurement_id"]
    source_record = json.loads((root / "docs/GPU_LANCZOS_SOURCE_IMPORT.json").read_text())
    current_sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in source_record["source_files"]}
    current_fingerprint = hashlib.sha256(json.dumps(current_sources, sort_keys=True).encode()).hexdigest()
    assert current_sources == source_record["source_files"]
    assert {row["source_sha256"] for row in runs} == {current_fingerprint}
    assert {r["validation_id"] for r in validation} == {measurement_id + "_validation"}
    previous = json.loads((root / "results/Vanda_eigen_1526107/comparison/provenance.json").read_text())
    previous_hashes = {g["model"]: g["sha256"] for g in previous["graphs"]}
    groups = defaultdict(list)
    for row in runs:
        assert row["graph_sha256"] == previous_hashes[row["model"]]
        assert int(row["spmv_count"]) == int(row["steps"]) + 1
        groups[(row["model"], int(row["budget"]), row["variant"])].append(row)
    assert len(groups) == 60
    index = {}
    for row in summary:
        key = (row["model"], int(row["budget"]), row["variant"])
        assert key not in index
        group = groups[key]
        assert len(group) == int(row["samples"]) == 5
        assert {r["repeat"] for r in group} == {"1", "2", "3", "4", "5"}
        for metric in ("solve_s", "iteration_s", "end_to_end_s"):
            values = [float(r[metric]) for r in group]
            assert all(math.isfinite(v) and v > 0 for v in values)
            for suffix, observed in (("median", statistics.median(values)), ("min", min(values)), ("max", max(values))):
                assert math.isclose(observed, float(row[f"{metric}_{suffix}"]), rel_tol=1e-12, abs_tol=1e-15)
        residuals = [float(r["direct_residual"]) for r in group]
        assert math.isclose(max(residuals), float(row["residual_max"]), rel_tol=1e-12, abs_tol=1e-18)
        assert (row["all_reached_tol"] == "True") == all(v < float(row["residual_tolerance"]) for v in residuals)
        index[key] = row
    selected = []
    for model in ("uniform", "hub", "community"):
        budget = min(b for b in (5, 10, 20, 30, 50)
                     if all(index[(model, b, v)]["all_reached_tol"] == "True" for v in variants))
        item = dict(model=model, budget=budget)
        for variant in sorted(variants):
            row = index[(model, budget, variant)]
            item[variant] = dict(solve_s=float(row["solve_s_median"]),
                                 end_to_end_s=float(row["end_to_end_s_median"]),
                                 residual_max=float(row["residual_max"]))
        cpu = item["cpu_mpi2_omp2"]
        device = item["gpu_mpi1"]
        item["solve_ratio_cpu2x2_over_gpu1"] = cpu["solve_s"] / device["solve_s"]
        item["application_ratio_cpu2x2_over_gpu1"] = cpu["end_to_end_s"] / device["end_to_end_s"]
        selected.append(item)
    assert {r["status"] for r in validation} == {"PASS"}
    assert Counter(r["variant"] for r in validation) == {v: 12 for v in variants}
    full = [r for r in validation if r["budget_tag"] == "full"]
    first = [r for r in validation if r["budget_tag"] == "one_step"]
    assert len(full) == len(first) == 24
    thresholds = runpy.run_path(str(root / "gpu_lanczos/scripts/lanczos_common.py"))["THRESHOLDS"]
    for row in validation:
        for field, threshold in (("norm_error", "norm_error"),
                                 ("residual_report_error", "dense_residual_agreement"),
                                 ("rayleigh_error", "rayleigh_agreement"),
                                 ("orthogonality", "orthogonality")):
            value = float(row[field])
            assert math.isfinite(value) and 0 <= value <= thresholds[threshold]
        if row["budget_tag"] == "full":
            expected = 0.5 if int(row["input_E"]) == 0 else 1.0
            assert abs(float(row["lambda_ritz"]) - expected) <= thresholds["eigenvalue_error"]
            assert float(row["dense_residual"]) <= thresholds["converged_residual"]
        else:
            assert float(row["one_step_L1"]) <= thresholds["one_step_L1"]
            assert float(row["one_step_Linf"]) <= thresholds["one_step_Linf"]
    output = dict(
        files={n: hashlib.sha256((folder / ("validation/comparisons.csv" if n == "comparisons.csv" else n)).read_bytes()).hexdigest()
               for n in ("runs.csv", "summary.csv", "comparisons.csv")},
        job_id="1536096.stdct-mgmt-02", hostname="GN-A40-003",
        measurement_id=measurement_id, performance_measurements=300, configurations=60,
        validation_records=48, validation_status="48 recorded PASS rows",
        graph_hashes_match_previous_eigenvalue_inputs=True,
        selected_common_passing_budgets=selected,
        validation_max_full_dense_residual=max(float(r["dense_residual"]) for r in full),
        validation_max_norm_error=max(float(r["norm_error"]) for r in validation),
        validation_max_one_step_L1=max(float(r["one_step_L1"]) for r in first),
        validation_max_residual_report_error=max(float(r["residual_report_error"]) for r in validation),
        scope="Existing CSV analysis only. Source fingerprint matches the imported implementation. No new solver execution or independent verification of original binaries, affinity logs, or vector dumps.",
        validation_thresholds_from_matching_source=thresholds,
        retained_validation_diagnostics_within_thresholds=True,
        imported_source_sha256=current_fingerprint,
        source_fingerprint_matches_measurements=True,
        solver_execution=False,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
