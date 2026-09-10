"""Compiler and relocatable-link primitives for the native-header contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence

try:
    from .model import CompileContext
except ImportError:  # Direct invocation through contract.py.
    from model import CompileContext  # type: ignore[no-redef]


class ProbeError(RuntimeError):
    """A generated probe could not be compiled or linked."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    output: str


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout: int = 120,
) -> ProcessResult:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ProcessResult(125, str(error))
    return ProcessResult(completed.returncode, completed.stdout or "")


def short_output(output: str, limit: int = 1800) -> str:
    value = output.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... output truncated ..."


def header_source(header: Path) -> str:
    include = f'#include "{header.as_posix()}"\n'
    return include + include


def compile_stdin(context: CompileContext, flags: Sequence[str], source: str) -> ProcessResult:
    args = (
        context.compiler,
        *flags,
        "-O0",
        "-fno-lto",
        "-fsyntax-only",
        "-x",
        "c++",
        "-",
    )
    return run(args, cwd=context.directory, input_text=source)


def compile_file(
    context: CompileContext,
    flags: Sequence[str],
    source: Path,
    output: Path,
) -> ProcessResult:
    args = (
        context.compiler,
        *flags,
        "-O0",
        "-fno-lto",
        "-x",
        "c++",
        "-c",
        str(source),
        "-o",
        str(output),
    )
    return run(args, cwd=context.directory)


def relocatable_link(
    context: CompileContext,
    flags: Sequence[str],
    objects: Sequence[Path],
    output: Path,
) -> ProcessResult:
    args = (
        context.compiler,
        *flags,
        "-O0",
        "-fno-lto",
        "-r",
        *(str(path) for path in objects),
        "-o",
        str(output),
    )
    return run(args, cwd=context.directory)


def write_odr_sources(directory: Path, headers: Sequence[Path]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    includes = "".join(header_source(header) for header in headers)
    reverse = "".join(header_source(header) for header in reversed(headers))
    first = directory / "forward.cpp"
    second = directory / "reverse.cpp"
    first.write_text(
        includes + '\nextern "C" int rund_native_header_marker_forward() { return 17; }\n',
        encoding="utf-8",
    )
    second.write_text(
        reverse + '\nextern "C" int rund_native_header_marker_reverse() { return 29; }\n',
        encoding="utf-8",
    )
    return first, second


def run_odr_probe(
    context: CompileContext,
    flags: Sequence[str],
    headers: Sequence[Path],
    directory: Path,
    label: str,
) -> None:
    sources = write_odr_sources(directory, headers)
    objects: list[Path] = []
    for source in sources:
        output = directory / (source.stem + ".o")
        result = compile_file(context, flags, source, output)
        if result.returncode != 0:
            raise ProbeError(
                f"{label} ODR compile failed for {source.name}: "
                f"{short_output(result.output)}"
            )
        objects.append(output)
    result = relocatable_link(context, flags, objects, directory / "combined.o")
    if result.returncode != 0:
        raise ProbeError(f"{label} ODR relocatable link failed: {short_output(result.output)}")
