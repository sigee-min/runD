set(RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT
  tests/contract/runtime/task/coroutine/allocation.cpp)

set(NODE_TEST_TASK_BASIC_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/basic.cpp
  tests/contract/runtime/task/lane.cpp)

set(NODE_TEST_TASK_HASH_TEST_SOURCES
  tests/contract/runtime/task/hash.cpp)

set(NODE_TEST_TASK_GROUP_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/group.cpp)

set(NODE_TEST_TASK_JOIN_ALL_TEST_SOURCES
  tests/contract/runtime/task/join/all.cpp)

set(NODE_TEST_TASK_CANCEL_TEST_SOURCES
  tests/contract/runtime/task/cancel.cpp)

set(NODE_TEST_TASK_LEAF_PARALLEL_TEST_SOURCES
  tests/contract/runtime/task/parallel/leaf.cpp)

set(NODE_TEST_TASK_COROUTINE_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/coroutine/cancellation.cpp
  tests/contract/runtime/task/coroutine/failure.cpp
  tests/contract/runtime/task/coroutine/lifecycle.cpp
  tests/contract/runtime/task/coroutine/lifecycle/basic.cpp
  tests/contract/runtime/task/coroutine/lifecycle/channel.cpp
  tests/contract/runtime/task/coroutine/lifecycle/assertion.cpp
  tests/contract/runtime/task/coroutine/lifecycle/io.cpp
  tests/contract/runtime/task/coroutine/lifecycle/io/many.cpp
  tests/contract/runtime/task/coroutine/lifecycle/task.cpp)

set(NODE_TEST_TASK_COROUTINE_FRAME_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/coroutine/frame.cpp)

set(NODE_TEST_TASK_RESULT_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/coroutine/result.cpp)

set(NODE_TEST_TASK_ENV_TEST_SOURCES
  ${RUND_NODE_TEST_TASK_ALLOCATION_SUPPORT}
  tests/contract/runtime/task/env.cpp)

set(NODE_TEST_TASK_RANDOM_TEST_SOURCES
  tests/contract/runtime/task/random.cpp)

set(NODE_TEST_TASK_READY_QUEUE_TEST_SOURCES
  tests/contract/runtime/task/ready/queue.cpp)

set(NODE_TEST_TASK_DEFAULT_RESOURCE_BUDGET_TEST_SOURCES
  tests/contract/runtime/task/resource/default.cpp)

set(NODE_TEST_TASK_RESOURCE_STATS_TEST_SOURCES
  tests/contract/runtime/task/resource/stats.cpp)
