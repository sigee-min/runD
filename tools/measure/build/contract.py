from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

from scan import (
    command,
    dependency_fanout,
    direct_include_rows,
    dirty_objects,
    live_objects,
    trace_paths,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    root = Path("/repo").resolve()
    build = root / ".cache/dev"
    live = live_objects(
        [
            "node/CMakeFiles/one.dir/src/one.cpp.o: CXX",
            "node/CMakeFiles/two.dir/tests/two.cpp.o: CXX",
            "node/not-an-object: phony",
        ]
    )
    require(len(live) == 2, "live object parser lost a graph output")
    roots = {
        "umbrella:task": (root / "node/include/rund/task.hpp").resolve(),
        "leaf:socketaccess": (
            root / "node/src/host/net/test/socket.hpp"
        ).resolve(),
        "leaf:computebackend": (
            root / "node/include/rund/compute/backend.hpp"
        ).resolve(),
        "leaf:computeops": (
            root / "node/include/rund/compute/ops.hpp"
        ).resolve(),
        "leaf:telemetryevent": (
            root / "node/include/rund/telemetry/event.hpp"
        ).resolve(),
        "leaf:testassert": (root / "tools/test/assert.hpp").resolve(),
    }
    fanout, recorded = dependency_fanout(
        [
            "node/CMakeFiles/one.dir/src/one.cpp.o: #deps 6, valid",
            "    ../../node/include/rund/task.hpp",
            "    ../../node/src/host/net/test/socket.hpp",
            "    ../../node/include/rund/compute/backend.hpp",
            "    ../../node/include/rund/compute/ops.hpp",
            "    ../../node/include/rund/telemetry/event.hpp",
            "    ../../tools/test/assert.hpp",
            "",
            "node/CMakeFiles/two.dir/tests/two.cpp.o: #deps 1, valid",
            "    ../../node/include/rund/task.hpp",
            "",
            "stale.o: #deps 1, valid",
            "    ../../node/include/rund/task.hpp",
        ],
        build,
        live,
        roots,
    )
    require(recorded == 2, "dependency parser admitted a stale output")
    require(fanout["umbrella:task"] == 2, "umbrella fan-out is incorrect")
    require(fanout["leaf:socketaccess"] == 1, "leaf fan-out is incorrect")
    require(
        fanout["leaf:computebackend"] == 1,
        "Compute backend leaf fan-out is incorrect",
    )
    require(
        fanout["leaf:computeops"] == 1,
        "Compute operation leaf fan-out is incorrect",
    )
    require(
        fanout["leaf:telemetryevent"] == 1,
        "telemetry event leaf fan-out is incorrect",
    )
    require(fanout["leaf:testassert"] == 1, "assertion fan-out is incorrect")

    require(
        direct_include_rows(
            [
                {
                    "command": "c++ -include /repo/node/src/host/net/test/socket.hpp -c one.cpp"
                },
                {"command": "c++ -c two.cpp"},
            ],
            roots["leaf:socketaccess"],
        )
        == 1,
        "direct include edge count is incorrect",
    )

    dirty = dirty_objects(
        [
            "[1/3] Building CXX object node/CMakeFiles/a.dir/src/a.cpp.o",
            "[2/3] Building CXX object node/CMakeFiles/b.dir/tests/b.cpp.o",
            "[3/3] Linking CXX executable node/test",
        ]
    )
    require(
        dirty == {"all": 2, "production": 1, "tests": 1, "other": 0},
        "dirty object classification is incorrect",
    )
    require(
        dirty_objects(["ninja: no work to do."])["production"] == 0,
        "no-op production dirtiness is not zero",
    )

    compiler, flags, directory = command(
        {
            "command": "c++ -DVALUE=1 -o out.o -MD -MF dep.d -c /src/main.cpp",
            "file": "/src/main.cpp",
            "directory": "/build",
        }
    )
    require(compiler == "c++", "compiler parser changed the driver")
    require(flags == ["-DVALUE=1"], "compiler parser retained output state")
    require(directory == Path("/build"), "compiler directory changed")
    require(
        trace_paths(b". /repo/a.hpp\n.. /sdk/b.hpp\n", Path("/build"))
        == {Path("/repo/a.hpp"), Path("/sdk/b.hpp")},
        "header trace parser is incorrect",
    )
    print("build measurement parser contract passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
