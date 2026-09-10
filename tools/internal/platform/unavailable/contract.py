"""Semantic and scaling contracts for the unavailable-platform audit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import verify


class AuditContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / ".cache/platform-unavailable"
        self.build.mkdir(parents=True)
        for relative in ("tools/internal/state/root", "tools/internal/state/roots.tsv"):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.cache = (
            "RUND_FORCE_UNAVAILABLE_PLATFORM:BOOL=ON\n"
            "RUND_ENABLE_VULKAN:BOOL=OFF\n"
            "RUND_TEST_NODE:BOOL=ON\n"
            "RUND_STRICT_WARNINGS:BOOL=ON\n"
            "RUND_NODE_FOCUSED_CASE:STRING=runtime.platform-adapter\n")
        self.rows = []
        for source, target in (
            ("src/runtime/platform/unavailable/a.cpp", "node-object-platform"),
            ("src/accel/vulkan/a.cpp", "node-object-accel-vulkan"),
            ("tests/contract/runtime/platform/adapter.cpp", "node-runtime"),
        ):
            self.rows.append({
                "directory": str(self.build),
                "file": str(self.root / "node" / source),
                "output": str(self.build / f"node/CMakeFiles/{target}.dir/{source}.o"),
                "command": "c++ -DRUND_NODE_PLATFORM_UNAVAILABLE=1 -c source.cpp",
            })
        for fragment, source in (
            ("platform.cmake", self.rows[0]["file"]),
            ("accel/vulkan.cmake", self.rows[1]["file"]),
        ):
            path = self.root / "node/cmake/node/sources" / fragment
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(Path(source).relative_to(self.root / "node")) + "\n")
        self.report = self.build / "audit.tsv"
        self.args = argparse.Namespace(
            root=self.root, build=self.build, phase="off-fresh",
            vulkan_enabled="OFF", objects_required="OFF", report=self.report)

    def run_audit(self) -> None:
        (self.build / "CMakeCache.txt").write_text(self.cache)
        (self.build / "compile_commands.json").write_text(json.dumps(self.rows))
        verify.audit(self.args)

    def reject(self, reason: str) -> None:
        before = self.report.read_bytes() if self.report.exists() else None
        with self.assertRaisesRegex(verify.AuditError, reason):
            self.run_audit()
        after = self.report.read_bytes() if self.report.exists() else None
        self.assertEqual(before, after, "a rejected audit published a partial phase")

    def build_objects(self) -> None:
        for entry in self.rows:
            path = Path(entry["directory"]) / entry["output"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture object")

    def test_off_on_off_transition(self) -> None:
        self.run_audit()
        self.build_objects()
        self.args.objects_required = "ON"
        self.args.vulkan_enabled = "ON"
        self.args.phase = "on-built"
        self.cache = self.cache.replace("RUND_ENABLE_VULKAN:BOOL=OFF", "RUND_ENABLE_VULKAN:BOOL=ON")
        self.rows[1]["command"] += " -DRUND_NODE_HAVE_VULKAN_SDK=1"
        self.run_audit()
        self.args.vulkan_enabled = "OFF"
        self.args.phase = "off-after-on"
        self.cache = self.cache.replace("RUND_ENABLE_VULKAN:BOOL=ON", "RUND_ENABLE_VULKAN:BOOL=OFF")
        self.rows[1]["command"] = self.rows[0]["command"]
        self.run_audit()
        rows = self.report.read_text().splitlines()
        self.assertEqual([row for row in rows if row.startswith("phase\t")],
                         ["phase\toff-fresh", "phase\ton-built", "phase\toff-after-on"])
        self.assertEqual(rows.count("platform_object_count\t1"), 2)
        self.assertEqual(rows.count("vulkan_sdk_definition\ttrue"), 1)

    def test_relative_database_paths(self) -> None:
        for row in self.rows:
            row["file"] = "../../" + str(Path(row["file"]).relative_to(self.root))
            row["output"] = str(Path(row["output"]).relative_to(self.build))
        self.build_objects()
        self.args.objects_required = "ON"
        self.run_audit()

    def test_missing_entry_fields(self) -> None:
        original = copy.deepcopy(self.rows)
        for field in ("directory", "file", "command", "output"):
            for value in (None, "", 7):
                with self.subTest(field=field, value=value):
                    self.rows = copy.deepcopy(original)
                    self.rows[0][field] = value
                    self.reject(f"nonempty {field}")

    def test_database_shape(self) -> None:
        self.rows = []
        self.reject("nonempty array")
        self.rows = [[]]
        self.reject("not an object")
        self.rows = {"file": "not an array"}
        self.reject("nonempty array")

    def test_phase_and_directory(self) -> None:
        self.args.phase = "../wrong"
        self.reject("phase is malformed")
        self.args.phase = "off-fresh"
        self.rows[0]["directory"] = "."
        self.reject("directory is not absolute")

    def test_target_escape(self) -> None:
        self.rows[0]["output"] = self.rows[0]["output"].replace(
            str(self.build), str(self.root / ".cache/foreign"))
        self.reject("canonical fragment")

    def test_exact_runtime_owner(self) -> None:
        original = copy.deepcopy(self.rows)
        self.rows.pop()
        self.reject("one compile owner")
        self.rows = copy.deepcopy(original) + [copy.deepcopy(original[-1])]
        self.reject("one compile owner")
        self.rows = copy.deepcopy(original)
        self.rows[-1]["output"] = self.rows[-1]["output"].replace("node-runtime.dir", "other.dir")
        self.reject("non-canonical target")

    def test_unavailable_body_and_fragment_ownership(self) -> None:
        original = copy.deepcopy(self.rows)
        for index in range(3):
            with self.subTest(index=index):
                self.rows = copy.deepcopy(original)
                self.rows[index]["command"] = "c++ -c source.cpp"
                self.reject("unavailable")
        self.rows = copy.deepcopy(original)
        self.rows[0]["file"] = str(self.root / "outside.cpp")
        self.reject("outside Node ownership")
        self.rows = copy.deepcopy(original)
        self.rows[1]["file"] = str(self.root / "node/src/accel/vulkan/unowned.cpp")
        self.reject("canonical fragment")

    def test_required_objects(self) -> None:
        self.args.objects_required = "ON"
        for index in range(3):
            with self.subTest(index=index):
                self.build_objects()
                Path(self.rows[index]["output"]).unlink()
                self.reject("not completely built|test object was not built")

    def test_sdk_definitions_in_any_entry(self) -> None:
        original = self.rows[-1]["command"]
        for define in ("RUND_NODE_HAVE_VULKAN_SDK", "RUND_NODE_GLSLANG_VALIDATOR", "RUND_NODE_SPIRV_VAL"):
            with self.subTest(define=define):
                self.rows[-1]["command"] = original + f" -D{define}=1"
                self.reject("Vulkan OFF retained")

    def test_cache_and_build_authority(self) -> None:
        original = self.cache
        self.cache += "RUND_STRICT_WARNINGS:BOOL=ON\n"
        self.reject("cache expected")
        self.cache = original.replace("runtime.platform-adapter", "runtime.task")
        self.reject("exact focused case")
        self.cache = original
        self.args.build = self.root / ".cache/other"
        self.reject("does not own build tree")

    def test_one_parse_for_large_database(self) -> None:
        extra = dict(self.rows[0], file=str(self.root / "unrelated.cpp"),
                     output=str(self.build / "unrelated.cpp.o"))
        self.rows += [extra] * 8000
        with mock.patch.object(verify.json, "loads", wraps=json.loads) as loads:
            self.run_audit()
        self.assertEqual(loads.call_count, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    ROOT, SCRATCH = args.root.resolve(), args.build.resolve()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__])
