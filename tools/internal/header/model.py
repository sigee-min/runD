"""Small, side-effect-free helpers for the native header contract.

The contract is intentionally driven by CMake's compile database.  This module
only parses rows and constructs compiler arguments; the runner owns temporary
files and subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Iterable, Sequence


class ModelError(ValueError):
    """A malformed or unusable compile-database contract input."""


NATIVE_DEFINE_PREFIXES = (
    "RUND_NODE_HAVE_VULKAN_SDK",
    "RUND_NODE_HAVE_GLSLANG_VALIDATOR",
    "RUND_NODE_HAVE_SPIRV_VAL",
    "RUND_NODE_HAVE_METAL_SDK",
    "RUND_NODE_GLSLANG_VALIDATOR",
    "RUND_NODE_SPIRV_VAL",
)

_PAIRED_OPTIONS = {
    "-o",
    "-MF",
    "-MT",
    "-MQ",
    "-MJ",
    "/Fo",
    "/Fd",
    "/Fa",
    "/Fe",
    "/Fp",
}
_JOINED_OUTPUT_PREFIXES = ("-o", "-MF", "-MT", "-MQ", "-MJ", "/Fo", "/Fd", "/Fa", "/Fe", "/Fp")


@dataclass(frozen=True)
class CompileRow:
    source: Path
    directory: Path
    compiler: str
    flags: tuple[str, ...]
    raw_file: str


@dataclass(frozen=True)
class CompileContext:
    kind: str
    source: Path
    directory: Path
    compiler: str
    flags: tuple[str, ...]


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _macro_name(word: str) -> str | None:
    if word.startswith("-D"):
        value = word[2:]
    elif word.startswith("/D"):
        value = word[2:]
    else:
        return None
    return value.split("=", 1)[0]


def is_native_define(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + "_") for prefix in NATIVE_DEFINE_PREFIXES)


def has_define(flags: Sequence[str], name: str) -> bool:
    index = 0
    while index < len(flags):
        word = flags[index]
        if word in {"-D", "/D"} and index + 1 < len(flags):
            if _macro_name("-D" + flags[index + 1]) == name:
                return True
            index += 2
            continue
        if _macro_name(word) == name:
            return True
        index += 1
    return False


def _resolve_word(word: str, directory: Path) -> Path:
    candidate = Path(word)
    if not candidate.is_absolute():
        candidate = directory / candidate
    return candidate.resolve()


def _is_source_word(word: str, source: Path, directory: Path, raw_file: str) -> bool:
    if word == raw_file or word == str(source):
        return True
    if word.startswith("-") or word.startswith("/") and len(word) > 1 and word[1].isalpha():
        return False
    try:
        return _resolve_word(word, directory) == source
    except OSError:
        return False


def clean_command_flags(
    words: Sequence[str],
    *,
    source: Path,
    directory: Path,
    raw_file: str,
) -> tuple[str, ...]:
    """Remove the anchor TU, outputs, dependency files, and build-only modes.

    Options are handled as argv entries, never by shell substitution.  Unknown
    options remain intact so include paths, language standards, and warning
    policy continue to come from the configured production context.
    """

    result: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in _PAIRED_OPTIONS:
            index += 2
            continue
        if any(word.startswith(prefix) and word != prefix for prefix in _JOINED_OUTPUT_PREFIXES):
            index += 1
            continue
        if word in {"-c", "/c", "-MD", "-MMD", "-MP", "-fsyntax-only"}:
            index += 1
            continue
        if word == "-x":
            index += 2
            continue
        if word in {"/TC", "/TP"}:
            index += 1
            continue
        if word.startswith("-O") or word.startswith("-flto") or word in {"-fno-lto", "/GL"}:
            index += 1
            continue
        if _is_source_word(word, source, directory, raw_file):
            index += 1
            continue
        result.append(word)
        index += 1
    return tuple(result)


def strip_native_defines(flags: Sequence[str]) -> tuple[str, ...]:
    """Return flags with SDK/validator feature definitions removed."""

    result: list[str] = []
    index = 0
    while index < len(flags):
        word = flags[index]
        if word in {"-D", "/D"} and index + 1 < len(flags):
            name = _macro_name("-D" + flags[index + 1])
            if name is not None and is_native_define(name):
                index += 2
                continue
            result.extend((word, flags[index + 1]))
            index += 2
            continue
        name = _macro_name(word)
        if name is not None and is_native_define(name):
            index += 1
            continue
        if word == "-framework" and index + 1 < len(flags):
            if flags[index + 1] in {"Metal", "Foundation", "Vulkan"}:
                index += 2
                continue
        result.append(word)
        index += 1
    return tuple(result)


def flags_for_mode(context: CompileContext, mode: str) -> tuple[str, ...]:
    """Produce deterministic C++ flags for ``off`` or ``on`` probing."""

    if mode == "off":
        return strip_native_defines(context.flags)
    if mode == "on":
        return context.flags
    raise ModelError(f"unknown SDK mode: {mode}")


def _words_from_row(row: dict[str, object], index: int) -> list[str]:
    command = row.get("command")
    arguments = row.get("arguments")
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError as error:
            raise ModelError(f"compile_commands[{index}] has invalid command quoting: {error}") from error
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        return [str(item) for item in arguments]
    raise ModelError(f"compile_commands[{index}] needs a command or arguments array")


def load_compile_database(path: Path, root: Path) -> tuple[CompileRow, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read compile database {path}: {error}") from error
    if not isinstance(data, list):
        raise ModelError(f"compile database {path} must contain an array")

    rows: list[CompileRow] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ModelError(f"compile_commands[{index}] is not an object")
        raw_file = item.get("file")
        raw_directory = item.get("directory", str(root))
        if not isinstance(raw_file, str) or not isinstance(raw_directory, str):
            raise ModelError(f"compile_commands[{index}] needs string file/directory fields")
        directory = Path(raw_directory)
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
        source = Path(raw_file)
        if not source.is_absolute():
            source = directory / source
        source = source.resolve()
        words = _words_from_row(item, index)
        if not words:
            raise ModelError(f"compile_commands[{index}] has an empty command")
        flags = clean_command_flags(
            words[1:], source=source, directory=directory, raw_file=raw_file
        )
        rows.append(
            CompileRow(
                source=source,
                directory=directory,
                compiler=words[0],
                flags=flags,
                raw_file=raw_file,
            )
        )
    return tuple(rows)


def _row_context(row: CompileRow, kind: str) -> CompileContext:
    return CompileContext(
        kind=kind,
        source=row.source,
        directory=row.directory,
        compiler=row.compiler,
        flags=row.flags,
    )


def select_context(rows: Iterable[CompileRow], root: Path, kind: str) -> CompileContext | None:
    """Select an anchor by source ownership, with SDK-enabled rows preferred."""

    node = root / "node" / "src"
    vulkan = node / "accel" / "vulkan"
    metal = node / "accel" / "metal"
    candidates: list[CompileRow] = []
    for row in rows:
        if row.source.suffix not in {".cpp", ".cc", ".cxx"}:
            continue
        if kind == "vulkan" and _under(row.source, vulkan):
            candidates.append(row)
        elif kind == "metal" and _under(row.source, metal):
            candidates.append(row)
        elif kind == "portable" and _under(row.source, node) and not _under(row.source, vulkan) and not _under(row.source, metal):
            candidates.append(row)
    if not candidates and kind == "portable":
        candidates = [row for row in rows if row.source.suffix in {".cpp", ".cc", ".cxx"}]
    if not candidates:
        return None

    def rank(row: CompileRow) -> tuple[int, int, str]:
        if kind == "vulkan":
            preferred = 0 if row.source.name == "capture.cpp" else 1
            sdk = 0 if has_define(row.flags, "RUND_NODE_HAVE_VULKAN_SDK") else 1
        elif kind == "metal":
            target = node / "accel" / "metal" / "kernel" / "pipeline" / "source" / "status" / "abi.cpp"
            preferred = 0 if row.source == target else (1 if row.source.name in {"abi.cpp", "source.cpp", "status.cpp"} else 2)
            sdk = 0 if has_define(row.flags, "RUND_NODE_HAVE_METAL_SDK") else 1
        else:
            preferred = 0
            sdk = 0
        return (sdk, preferred, str(row.source))

    return _row_context(sorted(candidates, key=rank)[0], kind)


def all_scoped_headers(root: Path) -> tuple[Path, ...]:
    vulkan_kernel = root / "node" / "src" / "accel" / "vulkan" / "kernel"
    explicit = (
        root / "node" / "src" / "accel" / "vulkan" / "command" / "capture.hpp",
        root / "node" / "src" / "accel" / "vulkan" / "command" / "dispatch.hpp",
        root / "node" / "src" / "accel" / "vulkan" / "command" / "timestamp.hpp",
        root / "node" / "src" / "accel" / "vulkan" / "buffer" / "create.hpp",
        root / "node" / "src" / "accel" / "metal" / "kernel" / "pipeline" / "abi.hpp",
    )
    if not vulkan_kernel.is_dir():
        raise ModelError(f"header contract scope root is missing: {vulkan_kernel}")
    headers = set(vulkan_kernel.rglob("*.hpp"))
    if not headers:
        raise ModelError(f"header contract scope root has no .hpp files: {vulkan_kernel}")
    headers.update(explicit)
    missing = sorted(path for path in headers if not path.is_file())
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise ModelError(f"header contract scope contains missing files: {names}")
    return tuple(sorted(path.resolve() for path in headers))


def sdk_available(context: CompileContext, kind: str) -> bool:
    # The configured component definition is authoritative.  A host SDK may
    # exist physically while this build deliberately compiled the component in
    # its SDK-off mode; probing that mode as SDK-on would hide the contract.
    return has_define(context.flags, "RUND_NODE_HAVE_" + kind.upper() + "_SDK")


def add_define(compiler: str, flags: Sequence[str], name: str) -> tuple[str, ...]:
    lower = Path(compiler).name.lower()
    option = "/D" if lower in {"cl", "cl.exe"} else "-D"
    return tuple(flags) + (option + name + "=1",)
