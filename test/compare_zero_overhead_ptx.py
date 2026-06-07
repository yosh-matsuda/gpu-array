#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run_nvcc(args: list[str]) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit(result.returncode)
    return result.stdout


def extract_entry(ptx: str, kernel_name: str) -> list[str]:
    # Extract only the selected kernel body so unrelated PTX metadata cannot affect the comparison.
    lines = ptx.splitlines()
    entry_index = None
    for index, line in enumerate(lines):
        if ".entry" in line and kernel_name in line:
            entry_index = index
            break

    if entry_index is None:
        raise ValueError(f"PTX entry for {kernel_name!r} was not found")

    while entry_index < len(lines) and "{" not in lines[entry_index]:
        entry_index += 1

    if entry_index == len(lines):
        raise ValueError(f"PTX entry body for {kernel_name!r} was not found")

    body: list[str] = []
    depth = 0
    for line in lines[entry_index:]:
        depth += line.count("{")
        depth -= line.count("}")
        if line.strip() not in {"{", "}"}:
            body.append(line)
        if depth == 0:
            break

    return body


def register_counts(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        clean_line = line.split("//", 1)[0].strip()
        match = re.match(r"\.reg\s+(\.\w+)\s+%\w+<(\d+)>;", clean_line)
        if match:
            counts[match.group(1)] = int(match.group(2))
    return counts


def opcode_sequence(lines: list[str]) -> list[str]:
    # Keep only instruction opcodes. Register names, labels, comments, and directives are compiler-noise here.
    opcodes: list[str] = []
    for line in lines:
        clean_line = line.split("//", 1)[0].strip()
        if not clean_line or clean_line.startswith(".") or clean_line.endswith(":"):
            continue
        if clean_line.startswith("{") or clean_line.startswith("}"):
            continue

        if clean_line.startswith("@"):
            parts = clean_line.split(None, 1)
            if len(parts) != 2:
                continue
            clean_line = parts[1]

        opcode = clean_line.split(None, 1)[0].rstrip(";")
        opcodes.append(opcode)
    return opcodes


def opcode_profile(lines: list[str]) -> Counter[str]:
    # For zip-style kernels, wrapper parameters change the prologue ordering. Ignore parameter plumbing and compare the
    # remaining instruction mix, which still catches extra arithmetic, branches, and memory operations in the loop body.
    ignored_opcodes = {
        "cvta.to.global.u64",
        "ld.param.u32",
        "ld.param.u64",
        "mov.u32",
        "mov.u64",
    }
    return Counter(opcode for opcode in opcode_sequence(lines) if opcode not in ignored_opcodes)


def format_sequence_diff(expected: list[str], actual: list[str]) -> str:
    limit = max(len(expected), len(actual))
    for index in range(limit):
        left = expected[index] if index < len(expected) else "<missing>"
        right = actual[index] if index < len(actual) else "<missing>"
        if left != right:
            start = max(0, index - 4)
            end = min(limit, index + 5)
            rows = [f"first differing opcode index: {index}"]
            for row_index in range(start, end):
                left_value = expected[row_index] if row_index < len(expected) else "<missing>"
                right_value = actual[row_index] if row_index < len(actual) else "<missing>"
                marker = "!=" if left_value != right_value else "=="
                rows.append(f"  {row_index:03d}: {left_value:<20} {marker} {right_value}")
            return "\n".join(rows)
    return "opcode sequences differ"


def ptxas_metrics(output: str, kernel_names: list[str]) -> dict[str, dict[str, int]]:
    # ptxas reports the post-PTX resource usage that matters for zero-overhead regressions.
    metrics: dict[str, dict[str, int]] = {}
    current_name = None
    for line in output.splitlines():
        if "Compiling entry function" in line:
            current_name = None
            for kernel_name in kernel_names:
                if kernel_name in line:
                    current_name = kernel_name
                    metrics[current_name] = {}
                    break
            continue

        if current_name is None:
            continue

        stack_match = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if stack_match:
            metrics[current_name]["stack_frame"] = int(stack_match.group(1))
            metrics[current_name]["spill_stores"] = int(stack_match.group(2))
            metrics[current_name]["spill_loads"] = int(stack_match.group(3))
            continue

        register_match = re.search(r"Used (\d+) registers", line)
        if register_match:
            metrics[current_name]["registers"] = int(register_match.group(1))

    missing = [kernel_name for kernel_name in kernel_names if kernel_name not in metrics]
    if missing:
        raise ValueError(f"ptxas metrics were not found for: {', '.join(missing)}")

    return metrics


def compare_ptx(
    reference_name: str,
    reference_lines: list[str],
    candidate_name: str,
    candidate_lines: list[str],
    compare_register_declarations: bool,
    opcode_mode: str,
) -> list[str]:
    errors: list[str] = []
    if compare_register_declarations:
        # This is intentionally optional: raw pointers and wrapper structs can need different PTX virtual registers for
        # argument unpacking even when ptxas emits the same final register usage.
        reference_registers = register_counts(reference_lines)
        candidate_registers = register_counts(candidate_lines)
        if candidate_registers != reference_registers:
            errors.append(
                f"{candidate_name}: register declarations differ from {reference_name}: "
                f"{candidate_registers} != {reference_registers}"
            )

    if opcode_mode == "sequence":
        reference_opcodes = opcode_sequence(reference_lines)
        candidate_opcodes = opcode_sequence(candidate_lines)
        if candidate_opcodes != reference_opcodes:
            errors.append(
                f"{candidate_name}: opcode sequence differs from {reference_name}\n"
                f"{format_sequence_diff(reference_opcodes, candidate_opcodes)}"
            )
    elif opcode_mode == "profile":
        reference_profile = opcode_profile(reference_lines)
        candidate_profile = opcode_profile(candidate_lines)
        if candidate_profile != reference_profile:
            errors.append(
                f"{candidate_name}: opcode profile differs from {reference_name}: "
                f"{dict(candidate_profile)} != {dict(reference_profile)}"
            )
    else:
        raise ValueError(f"unknown opcode comparison mode: {opcode_mode}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare zero-overhead CUDA PTX kernels after stable normalization.")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--host-compiler", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--include-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ptx_path = output_dir / "zero_overhead.ptx"
    cubin_path = output_dir / "zero_overhead.cubin"
    arch = args.arch.removeprefix("sm_").removeprefix("compute_")

    ptx_command = [
        args.compiler,
        "--ptx",
        f"--gpu-architecture=compute_{arch}",
        "--std=c++20",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
        # Measure release-like abstraction overhead, not debug assertion paths in wrapper dereference operators.
        "-DNDEBUG",
        "-ccbin",
        args.host_compiler,
        "-I",
        args.include_dir,
        "-o",
        str(ptx_path),
        args.source,
    ]
    run_nvcc(ptx_command)

    cubin_command = [
        args.compiler,
        "--cubin",
        f"--gpu-architecture=sm_{arch}",
        "--std=c++20",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
        # Keep ptxas metrics aligned with the PTX generated above.
        "-DNDEBUG",
        "-Xptxas=-v",
        "-ccbin",
        args.host_compiler,
        "-I",
        args.include_dir,
        "-o",
        str(cubin_path),
        args.source,
    ]
    ptxas_output = run_nvcc(cubin_command)

    ptx = ptx_path.read_text(encoding="utf-8")
    # Each group compares a raw-pointer baseline against equivalent gpu-array abstractions. Simpler kernels use strict
    # opcode sequencing; zip kernels use an opcode profile because their argument-unpacking prologues differ naturally.
    groups = {
        "block-thread stride": {
            "opcode_mode": "sequence",
            "kernels": [
                ("raw pointer", "ga_ptx_raw_pointer_kernel", False),
                ("managed_array", "ga_ptx_array_kernel", False),
                ("managed_array view", "ga_ptx_array_view_kernel", False),
            ],
        },
        "enumerate grid-thread stride": {
            "opcode_mode": "sequence",
            "kernels": [
                ("raw pointer", "ga_ptx_raw_enumerate_kernel", False),
                ("enumerate view", "ga_ptx_enumerate_view_kernel", False),
            ],
        },
        "grid-thread stride": {
            "opcode_mode": "sequence",
            "kernels": [
                ("raw pointer", "ga_ptx_raw_grid_stride_kernel", False),
                ("grid-thread stride view", "ga_ptx_grid_stride_view_kernel", False),
            ],
        },
        "zip grid-thread stride": {
            "opcode_mode": "profile",
            "kernels": [
                ("raw pointer", "ga_ptx_raw_zip_kernel", False),
                ("zip view", "ga_ptx_zip_view_kernel", False),
            ],
        },
        "zip enumerate grid-thread stride": {
            "opcode_mode": "profile",
            "kernels": [
                ("raw pointer", "ga_ptx_raw_zip_enumerate_kernel", False),
                ("zip then enumerate view", "ga_ptx_zip_enumerate_view_kernel", False),
            ],
        },
        "single-value dereference": {
            "opcode_mode": "profile",
            "kernels": [
                ("raw pointer", "ga_ptx_raw_value_kernel", False),
                ("value", "ga_ptx_value_kernel", False),
                ("managed_value", "ga_ptx_managed_value_kernel", False),
            ],
        },
    }
    kernel_names = {
        f"{group_name} / {display_name}": kernel_name
        for group_name, group_config in groups.items()
        for display_name, kernel_name, _ in group_config["kernels"]
    }
    kernels = {key: extract_entry(ptx, kernel_name) for key, kernel_name in kernel_names.items()}
    resources = ptxas_metrics(ptxas_output, list(kernel_names.values()))

    errors = []
    for group_name, group_config in groups.items():
        group = group_config["kernels"]
        reference_display_name, reference_kernel_name, _ = group[0]
        reference_key = f"{group_name} / {reference_display_name}"
        for display_name, kernel_name, compare_register_declarations in group[1:]:
            candidate_key = f"{group_name} / {display_name}"
            errors.extend(
                compare_ptx(
                    reference_key,
                    kernels[reference_key],
                    candidate_key,
                    kernels[candidate_key],
                    compare_register_declarations,
                    group_config["opcode_mode"],
                )
            )
            if resources[kernel_name] != resources[reference_kernel_name]:
                errors.append(
                    f"{group_name} / {display_name}: ptxas resources differ from {reference_display_name}: "
                    f"{resources[kernel_name]} != {resources[reference_kernel_name]}"
                )

    if errors:
        print("Zero-overhead PTX comparison failed:")
        for error in errors:
            print(f"\n{error}")
        print(f"\nGenerated PTX: {ptx_path}")
        return 1

    print("Zero-overhead PTX comparison passed:")
    for group_name, group_config in groups.items():
        print(f"  {group_name}:")
        for name, kernel_name, _ in group_config["kernels"]:
            lines = kernels[f"{group_name} / {name}"]
            print(
                f"    {name}: ptx_registers={register_counts(lines)}, "
                f"ptx_opcodes={len(opcode_sequence(lines))}, "
                f"ptx_profile={dict(opcode_profile(lines))}, ptxas={resources[kernel_name]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
