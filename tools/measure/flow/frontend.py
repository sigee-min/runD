#!/usr/bin/env python3
import csv
import statistics
import subprocess
import sys
import time


def sample(compiler: str, include: str, source: str) -> list[float]:
    command = [
        compiler,
        "-std=c++20",
        f"-I{include}",
        "-fsyntax-only",
        source,
    ]
    subprocess.run(command, check=True, timeout=120)
    values = []
    for _ in range(7):
        begin = time.perf_counter_ns()
        subprocess.run(command, check=True, timeout=120)
        values.append((time.perf_counter_ns() - begin) / 1_000_000.0)
    return values


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: frontend.py <compiler> <include> <minimal> <typed> <output>",
            file=sys.stderr,
        )
        return 2
    compiler, include, minimal, typed, output = sys.argv[1:]
    rows = []
    for name, source in (("frontend_minimal", minimal), ("frontend_typed", typed)):
        values = sample(compiler, include, source)
        rows.extend((name, index + 1, "ms", value) for index, value in enumerate(values))
        rows.append((name, "median", "ms", statistics.median(values)))
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "sample", "unit", "value"))
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
