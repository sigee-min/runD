"""Audit the unavailable-platform compile owners from one parsed database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def fragment_sources(path: Path, pattern: str) -> set[str]:
    sources = set(re.findall(pattern, path.read_text()))
    require(bool(sources), f"platform source fragment has no matching sources: {path}")
    return sources


def audit(args: argparse.Namespace) -> None:
    root, build = args.root.resolve(), args.build.resolve()
    expected_build = subprocess.check_output(
        ["sh", str(root / "tools/internal/state/root"), str(root),
         "platform-unavailable"], text=True).strip()
    require(str(build) == expected_build,
            f"platform-unavailable verification does not own build tree: {build}")
    require(re.fullmatch(r"[a-z][a-z0-9-]*", args.phase) is not None,
            "platform-unavailable phase is malformed")

    cache = (build / "CMakeCache.txt").read_text().splitlines()
    for name, value in {
        "RUND_FORCE_UNAVAILABLE_PLATFORM": "ON",
        "RUND_ENABLE_VULKAN": args.vulkan_enabled,
        "RUND_TEST_NODE": "ON",
        "RUND_STRICT_WARNINGS": "ON",
    }.items():
        rows = [row for row in cache if row.startswith(name + ":BOOL=")]
        require(rows == [f"{name}:BOOL={value}"],
                f"platform-unavailable cache expected {name}:BOOL={value}, got {rows}")
    focus = [row for row in cache
             if re.match(r"^RUND_NODE_FOCUSED_CASE:[A-Z]+=", row)]
    require(len(focus) == 1 and re.fullmatch(
        r"RUND_NODE_FOCUSED_CASE:[A-Z]+=runtime\.platform-adapter", focus[0])
        is not None, f"platform-unavailable cache lost the exact focused case: {focus}")

    node = root / "node"
    expected = {
        "platform": fragment_sources(
            node / "cmake/node/sources/platform.cmake",
            r"src/runtime/platform/unavailable/[A-Za-z0-9_./-]+\.cpp"),
        "vulkan": fragment_sources(
            node / "cmake/node/sources/accel/vulkan.cmake",
            r"src/accel/vulkan/[A-Za-z0-9_./-]+\.cpp"),
    }
    # Repeated CMake string(JSON GET) reparsed this entire document for each
    # field of every entry. One parse followed by one traversal is O(bytes).
    with (build / "compile_commands.json").open() as stream:
        commands = json.load(stream)
    require(isinstance(commands, list) and bool(commands),
            "platform-unavailable compile database must be a nonempty array")
    actual: dict[str, set[str]] = {owner: set() for owner in expected}
    objects = dict.fromkeys(expected, 0)
    runtime_entries = runtime_objects = 0
    definitions = {
        "vulkan_sdk_definition": re.compile(r"RUND_NODE_HAVE_VULKAN_SDK"),
        "glslang_definition": re.compile(r"RUND_NODE_[A-Z0-9_]*GLSLANG[A-Z0-9_]*"),
        "spirv_definition": re.compile(r"RUND_NODE_[A-Z0-9_]*SPIRV[A-Z0-9_]*"),
    }
    found = dict.fromkeys(definitions, False)
    targets = {
        "platform": build / "node/CMakeFiles/node-object-platform.dir",
        "vulkan": build / "node/CMakeFiles/node-object-accel-vulkan.dir",
    }
    runtime_source = node / "tests/contract/runtime/platform/adapter.cpp"
    runtime_output = (build / "node/CMakeFiles/node-runtime.dir/tests/contract/"
                      "runtime/platform/adapter.cpp.o")

    for index, entry in enumerate(commands):
        require(isinstance(entry, dict), f"compile entry {index} is not an object")
        for field in ("directory", "file", "command", "output"):
            require(isinstance(entry.get(field), str) and bool(entry[field]),
                    f"compile entry {index} has no nonempty {field}")
        directory = Path(entry["directory"])
        require(directory.is_absolute(), f"compile entry {index} directory is not absolute")
        source = (directory / entry["file"]).resolve()
        output = (directory / entry["output"]).resolve()
        command = entry["command"]
        for name, pattern in definitions.items():
            found[name] = found[name] or pattern.search(command) is not None

        if source == runtime_source:
            runtime_entries += 1
            require(output == runtime_output,
                    f"platform-adapter case has a non-canonical target: {output}")
            require("RUND_NODE_PLATFORM_UNAVAILABLE=1" in command,
                    "platform-adapter case would not execute its unavailable-owner body")
            runtime_objects += output.is_file()

        owner = next((name for name, target in targets.items()
                      if output.is_relative_to(target)), None)
        if owner is None:
            continue
        require("RUND_NODE_PLATFORM_UNAVAILABLE=1" in command,
                f"{owner} object is missing unavailable-platform selection: {source}")
        require(source.is_relative_to(node),
                f"{owner} object source is outside Node ownership: {source}")
        actual[owner].add(source.relative_to(node).as_posix())
        objects[owner] += output.is_file()

    for owner in expected:
        require(actual[owner] == expected[owner],
                f"{owner} object sources disagree with the canonical fragment: "
                f"expected={sorted(expected[owner])}, actual={sorted(actual[owner])}")
    require(runtime_entries == 1,
            f"platform-adapter exact test must have one compile owner: {runtime_entries}")
    if args.objects_required == "ON":
        for owner in expected:
            require(objects[owner] == len(expected[owner]),
                    f"{owner} object owner was not completely built: "
                    f"{objects[owner]}/{len(expected[owner])}")
        require(runtime_objects == 1, "platform-adapter exact test object was not built")
    require(args.vulkan_enabled == "ON" or not any(found.values()),
            "Vulkan OFF retained SDK, glslang, or SPIR-V compile definitions")

    rows = {
        "phase": args.phase,
        "vulkan_enabled": args.vulkan_enabled,
        "objects_required": args.objects_required,
        "platform_source_count": len(expected["platform"]),
        "platform_object_count": objects["platform"],
        "vulkan_source_count": len(expected["vulkan"]),
        "vulkan_object_count": objects["vulkan"],
        "runtime_test_entry_count": runtime_entries,
        "runtime_test_object_count": runtime_objects,
        **{name: str(value).lower() for name, value in found.items()},
    }
    with args.report.open("a") as stream:
        stream.writelines(f"{name}\t{value}\n" for name, value in rows.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "build", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--vulkan-enabled", choices=("ON", "OFF"), required=True)
    parser.add_argument("--objects-required", choices=("ON", "OFF"), required=True)
    try:
        audit(parser.parse_args())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"platform-unavailable verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
