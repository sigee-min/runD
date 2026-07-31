list(APPEND NODE_SOURCES
  src/runtime/task/scheduler/task/commit.cpp
  src/runtime/task/scheduler/task/primitive.cpp
  src/runtime/task/scheduler/task/context/run.cpp
  src/runtime/task/scheduler/task/context/switch/coroutine.cpp
  src/runtime/task/scheduler/task/context/switch/resume.cpp
  src/runtime/task/scheduler/task/context/switch/leaf.cpp
  src/runtime/task/scheduler/task/context/switch.cpp
  src/runtime/task/scheduler/task/context/fail.cpp
  src/runtime/task/scheduler/task/direct.cpp
  src/runtime/task/scheduler/task/external/dispatch.cpp
  src/runtime/task/scheduler/task/external/queue.cpp
  src/runtime/task/scheduler/task/external/park.cpp
  src/runtime/task/scheduler/task/external/wake.cpp
  src/runtime/task/scheduler/task/retire.cpp
)
