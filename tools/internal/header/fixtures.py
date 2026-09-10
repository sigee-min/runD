"""Compiler-only negative/positive controls for the native-header contract."""

from __future__ import annotations

from pathlib import Path

try:
    from .model import CompileContext, add_define, strip_native_defines
    from .probe import compile_file, compile_stdin, relocatable_link, short_output
except ImportError:  # Direct invocation through contract.py.
    from model import CompileContext, add_define, strip_native_defines  # type: ignore[no-redef]
    from probe import compile_file, compile_stdin, relocatable_link, short_output  # type: ignore[no-redef]


class FixtureError(RuntimeError):
    """A self-test control did not exhibit its expected compiler result."""


def _require_failure(context: CompileContext, flags: tuple[str, ...], source: str, label: str) -> None:
    result = compile_stdin(context, flags, source)
    if result.returncode == 0:
        raise FixtureError(f"self-test {label} fixture was accepted")


def _require_success(context: CompileContext, flags: tuple[str, ...], source: str, label: str) -> None:
    result = compile_stdin(context, flags, source)
    if result.returncode != 0:
        raise FixtureError(f"self-test {label} positive control failed: {short_output(result.output)}")


def run_self_tests(context: CompileContext, directory: Path) -> None:
    """Exercise fail-closed fixture paths without running generated code."""

    directory.mkdir(parents=True, exist_ok=True)
    flags = strip_native_defines(context.flags)

    missing_include = (
        "int rund_native_header_missing_include_fixture() {\n"
        "  std::uint32_t value = 0;\n"
        "  return static_cast<int>(value);\n"
        "}\n"
    )
    _require_failure(context, flags, missing_include, "missing-include")
    _require_success(context, flags, "#include <cstdint>\n" + missing_include, "missing-include")

    sdk_only = (
        "#if !defined(RUND_NATIVE_HEADER_SDK_SELFTEST)\n"
        '#error "SDK-only fixture must fail when the SDK definition is absent"\n'
        "#endif\n"
        "int rund_native_header_sdk_fixture() { return 1; }\n"
    )
    _require_failure(context, flags, sdk_only, "SDK-only")
    _require_success(
        context,
        add_define(context.compiler, flags, "RUND_NATIVE_HEADER_SDK_SELFTEST"),
        sdk_only,
        "SDK-only",
    )

    duplicate_header = directory / "fixture-duplicate-definition.hpp"
    duplicate_header.write_text(
        "int rund_native_header_duplicate_definition() { return 7; }\n",
        encoding="utf-8",
    )
    first = directory / "fixture-duplicate-first.cpp"
    second = directory / "fixture-duplicate-second.cpp"
    first.write_text(
        f'#include "{duplicate_header.as_posix()}"\n'
        'extern "C" int rund_native_header_fixture_first() { return 1; }\n',
        encoding="utf-8",
    )
    second.write_text(
        f'#include "{duplicate_header.as_posix()}"\n'
        'extern "C" int rund_native_header_fixture_second() { return 2; }\n',
        encoding="utf-8",
    )
    first_object = directory / "fixture-duplicate-first.o"
    second_object = directory / "fixture-duplicate-second.o"
    for source, output in ((first, first_object), (second, second_object)):
        result = compile_file(context, flags, source, output)
        if result.returncode != 0:
            raise FixtureError(f"self-test duplicate fixture failed to compile: {short_output(result.output)}")

    positive = relocatable_link(
        context,
        flags,
        (first_object,),
        directory / "fixture-duplicate-positive.o",
    )
    if positive.returncode != 0:
        raise FixtureError(f"self-test single-object link positive control failed: {short_output(positive.output)}")
    negative = relocatable_link(
        context,
        flags,
        (first_object, second_object),
        directory / "fixture-duplicate-negative.o",
    )
    if negative.returncode == 0:
        raise FixtureError("self-test non-inline duplicate-definition fixture was accepted")
