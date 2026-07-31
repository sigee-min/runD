from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable


HEADERS = (
    "task",
    "session",
    "net",
    "replay",
    "compute",
    "compute/async",
    "compute/math",
    "compute/pipeline",
    "compute/session",
)
ANCHOR = "node/tests/contract/main.cpp"
TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
OBJECT = re.compile(r"^(.+[.](?:o|obj)): #deps [0-9]+,")
TRACE = re.compile(r"^[.]+ (.+)$")
DIRTY = re.compile(r"Building (?:C|CXX|OBJCXX) object (.+[.](?:o|obj))$")


def fail(message: str) -> None:
    raise RuntimeError(message)


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def command(row: dict[str, object]) -> tuple[str, list[str], Path]:
    text = row.get("command")
    source = row.get("file")
    directory = row.get("directory")
    if not isinstance(text, str) or not isinstance(source, str) or not isinstance(
        directory, str
    ):
        fail("compile database anchor must own command, file, and directory strings")
    words = shlex.split(text)
    if not words:
        fail("compile database anchor has an empty command")

    compiler = words[0]
    flags: list[str] = []
    skip = False
    paired = {"-o", "-MF", "-MT", "-MQ"}
    standalone = {"-c", "-MD", "-MMD", "-MP"}
    for word in words[1:]:
        if skip:
            skip = False
            continue
        if word in paired:
            skip = True
            continue
        if word in standalone or word == source:
            continue
        flags.append(word)
    if skip:
        fail("compile database anchor ends inside a paired compiler option")
    return compiler, flags, Path(directory).resolve()


def anchor(database: Path, root: Path) -> tuple[str, list[str], Path]:
    try:
        rows = json.loads(database.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"compile database is unreadable: {error}")
    if not isinstance(rows, list):
        fail("compile database root must be a list")
    expected = (root / ANCHOR).resolve()
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("file"), str)
        and Path(str(row["file"])).resolve() == expected
    ]
    if len(matches) != 1:
        fail(f"compile database must contain one canonical anchor: {ANCHOR}")
    return command(matches[0])


def execute(
    args: list[str], directory: Path, source: bytes, *, stderr: bool = False
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        args,
        cwd=directory,
        input=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.decode(errors="replace")
        fail(f"compiler probe failed: {' '.join(args)}\n{diagnostic}")
    if stderr:
        return result
    return result


def trace_paths(output: bytes, directory: Path) -> set[Path]:
    paths: set[Path] = set()
    for line in output.decode(errors="replace").splitlines():
        match = TRACE.match(line)
        if match is None:
            continue
        raw = match.group(1).removesuffix(" (framework directory)")
        path = Path(raw)
        paths.add((directory / path).resolve() if not path.is_absolute() else path.resolve())
    return paths


def probe(
    compiler: str,
    flags: list[str],
    directory: Path,
    root: Path,
    header: str,
    samples: int,
) -> tuple[dict[str, int], set[Path]]:
    source = f"#include <rund/{header}.hpp>\n".encode()
    prefix = [compiler, *flags]
    syntax = [*prefix, "-fsyntax-only", "-x", "c++", "-"]
    cold_start = time.perf_counter_ns()
    execute(syntax, directory, source)
    cold_duration = time.perf_counter_ns() - cold_start
    preprocessed = execute(
        [*prefix, "-E", "-P", "-x", "c++", "-"], directory, source
    ).stdout
    traced = execute(
        [*prefix, "-H", "-fsyntax-only", "-x", "c++", "-"],
        directory,
        source,
        stderr=True,
    )
    paths = trace_paths(traced.stderr, directory)

    durations: list[int] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        execute(syntax, directory, source)
        durations.append(time.perf_counter_ns() - start)

    return (
        {
            "preprocessed_bytes": len(preprocessed),
            "local_headers": sum(inside(path, root) for path in paths),
            "transitive_headers": len(paths),
            "syntax_cold_ns": cold_duration,
            "syntax_warm_median_ns": int(statistics.median(durations)),
        },
        paths,
    )


def ninja(root: Path, build: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["sh", str(root / "tools/internal/state/ninja"), "-C", str(build), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        fail(f"Ninja query failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout.splitlines()


def ninja_stream(root: Path, build: Path, *args: str) -> Iterable[str]:
    process = subprocess.Popen(
        ["sh", str(root / "tools/internal/state/ninja"), "-C", str(build), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        fail("Ninja stream did not expose its output pipes")
    for line in process.stdout:
        yield line.rstrip("\n")
    diagnostic = process.stderr.read()
    if process.wait() != 0:
        fail(f"Ninja query failed: {' '.join(args)}\n{diagnostic}")


def live_objects(lines: Iterable[str]) -> set[str]:
    objects: set[str] = set()
    for line in lines:
        output, separator, _ = line.partition(": ")
        if separator and output.endswith((".o", ".obj")):
            objects.add(output)
    return objects


def dependency_fanout(
    lines: Iterable[str], build: Path, live: set[str], roots: dict[str, Path]
) -> tuple[dict[str, int], int]:
    found: dict[str, set[str]] = {name: set() for name in roots}
    spellings: dict[str, str] = {}
    for name, path in roots.items():
        for spelling in (str(path), os.path.relpath(path, build)):
            prior = spellings.setdefault(spelling, name)
            if prior != name:
                fail(f"dependency spelling aliases two owners: {spelling}")
    recorded: set[str] = set()
    current = ""
    for line in lines:
        match = OBJECT.match(line)
        if match is not None:
            current = match.group(1)
            if current in live:
                recorded.add(current)
            continue
        if current not in live or not line[:1].isspace():
            continue
        raw = line.strip()
        name = spellings.get(raw)
        if name is not None:
            found[name].add(current)
    return {name: len(objects) for name, objects in found.items()}, len(recorded)


def direct_include_rows(rows: object, header: Path) -> int:
    if not isinstance(rows, list):
        fail("compile database root must be a list")
    expected = str(header)
    count = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("command"), str):
            fail("compile database rows must own command strings")
        words = shlex.split(str(row["command"]))
        includes = [
            words[index + 1]
            for index, word in enumerate(words[:-1])
            if word == "-include"
        ]
        count += expected in includes
    return count


def direct_include_count(database: Path, header: Path) -> int:
    try:
        rows = json.loads(database.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"compile database is unreadable: {error}")
    return direct_include_rows(rows, header)


def dirty_objects(lines: Iterable[str]) -> dict[str, int]:
    objects: set[str] = set()
    for line in lines:
        match = DIRTY.search(line)
        if match is not None:
            objects.add(match.group(1))
    production = sum("/src/" in path for path in objects)
    tests = sum("/tests/" in path for path in objects)
    return {
        "all": len(objects),
        "production": production,
        "tests": tests,
        "other": len(objects) - production - tests,
    }


def manifest(root: Path, output: Path) -> bytes:
    result = subprocess.run(
        ["sh", str(root / "tools/source/manifest"), str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        fail(result.stderr.decode(errors="replace"))
    return output.read_bytes()


def measure(root: Path, build: Path, target: str, samples: int) -> list[str]:
    if not inside(build, root / ".cache"):
        fail("build measurement requires a repository-owned .cache tree")
    if not TARGET.fullmatch(target):
        fail(f"invalid Ninja target: {target}")
    if samples < 3 or samples % 2 == 0:
        fail("syntax sample count must be odd and at least three")
    database = build / "compile_commands.json"
    if not (build / "build.ninja").is_file() or not database.is_file():
        fail(f"build tree lacks Ninja or compile database authority: {build}")

    cache = root / ".cache/measure/build"
    cache.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
    try:
        before = manifest(root, scratch / "before.tsv")
        compiler, flags, directory = anchor(database, root)
        compiler_path = Path(compiler).resolve()
        flag_hash = hashlib.sha256(
            b"\0".join(
                [str(compiler_path).encode(), str(directory).encode()]
                + [flag.encode() for flag in flags]
            )
        ).hexdigest()
        version = subprocess.run(
            [compiler, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        ).stdout.splitlines()[0]

        header_results = {
            header: probe(compiler, flags, directory, root, header, samples)
            for header in HEADERS
        }
        header_metrics = {
            header: result[0] for header, result in header_results.items()
        }
        header_paths = {
            header: result[1] for header, result in header_results.items()
        }
        pipeline_header = (
            root / "node/include/rund/compute/pipeline.hpp"
        ).resolve()
        compute_pipeline_reach = int(pipeline_header in header_paths["compute"])
        if compute_pipeline_reach != 0:
            fail("basic Compute header imports the opt-in Pipeline surface")
        live = live_objects(ninja(root, build, "-t", "targets", "all"))
        roots = {
            **{
                f"umbrella:{header}": (
                    root / f"node/include/rund/{header}.hpp"
                ).resolve()
                for header in HEADERS
            },
            "leaf:socketaccess": (
                root / "node/src/host/net/test/socket.hpp"
            ).resolve(),
            "leaf:sessioncompute": (
                root / "node/include/rund/compute/session/await.hpp"
            ).resolve(),
            "leaf:computebackend": (
                root / "node/include/rund/compute/backend.hpp"
            ).resolve(),
            "leaf:computeops": (
                root / "node/include/rund/compute/ops.hpp"
            ).resolve(),
            **{
                f"leaf:computeabi:{name}": (
                    root / f"node/include/rund/compute/abi/{name}.hpp"
                ).resolve()
                for name in (
                    "model",
                    "device",
                    "expression",
                    "job",
                    "observe",
                    "graph",
                    "flow",
                )
            },
            "leaf:telemetryevent": (
                root / "node/include/rund/telemetry/event.hpp"
            ).resolve(),
            "leaf:testassert": (root / "tools/test/assert.hpp").resolve(),
            "leaf:metalcachedindex": (
                root / "node/src/accel/metal/pipeline/artifact/index.hpp"
            ).resolve(),
            "leaf:vulkancachedindex": (
                root / "node/src/accel/vulkan/cached/index.hpp"
            ).resolve(),
        }
        fanout, dependency_objects = dependency_fanout(
            ninja_stream(root, build, "-t", "deps"), build, live, roots
        )
        socket_direct = direct_include_count(database, roots["leaf:socketaccess"])
        dirty = dirty_objects(ninja(root, build, "-n", target))
        after = manifest(root, scratch / "after.tsv")
        if before != after:
            fail("product source changed during build measurement")

        rows = [
            f"context\tmanifest\t{hashlib.sha256(before).hexdigest()}\tsha256",
            f"context\tcompiler\t{compiler_path}\tpath",
            f"context\tcompiler_version\t{version}\ttext",
            f"context\tflags\t{flag_hash}\tsha256",
            f"context\tanchor\t{ANCHOR}\tpath",
            f"context\tsamples\t{samples}\tcount",
            f"graph\tobjects\t{len(live)}\tobject",
            f"graph\tdependency_objects\t{dependency_objects}\tobject",
            f"edge:leaf:socketaccess\tdirect_commands\t{socket_direct}\tcommand",
            "edge:umbrella:compute/pipeline\ttransitive_reach\t"
            f"{compute_pipeline_reach}\theader",
        ]
        for header in HEADERS:
            for metric in (
                "preprocessed_bytes",
                "local_headers",
                "transitive_headers",
                "syntax_cold_ns",
                "syntax_warm_median_ns",
            ):
                unit = (
                    "byte"
                    if metric == "preprocessed_bytes"
                    else "ns"
                    if metric in ("syntax_cold_ns", "syntax_warm_median_ns")
                    else "header"
                )
                rows.append(
                    f"header:{header}\t{metric}\t{header_metrics[header][metric]}\t{unit}"
                )
        for name in [
            *(f"umbrella:{header}" for header in HEADERS),
            "leaf:socketaccess",
            "leaf:sessioncompute",
            "leaf:computebackend",
            "leaf:computeops",
            "leaf:computeabi:model",
            "leaf:computeabi:device",
            "leaf:computeabi:expression",
            "leaf:computeabi:job",
            "leaf:computeabi:observe",
            "leaf:computeabi:graph",
            "leaf:computeabi:flow",
            "leaf:telemetryevent",
            "leaf:testassert",
            "leaf:metalcachedindex",
            "leaf:vulkancachedindex",
        ]:
            rows.append(
                f"fanout:{name}\tmaterialized_objects\t{fanout[name]}\tobject"
            )
        for kind in ("all", "production", "tests", "other"):
            rows.append(f"dirty:{target}\t{kind}\t{dirty[kind]}\tobject")
        return rows
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print("usage: scan.py <root> <build> <target> <samples>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    build = Path(argv[2]).resolve()
    try:
        samples = int(argv[4])
        rows = measure(root, build, argv[3], samples)
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"build measurement failed: {error}", file=sys.stderr)
        return 1
    print("scope\tmetric\tvalue\tunit")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
