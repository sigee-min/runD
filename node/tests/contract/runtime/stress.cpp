#include "test/assert.hpp"

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/program/executor.hpp>
#include <kernel/program/executor/policy.hpp>
#include <kernel/program/skeleton.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace {

rund::task::Task<void>
ProbeNestedParallel(std::array<rund::kernel::i32, 96u> *const storage,
                    rund::kernel::SkeletonResult *const result) {
  (void)co_await rund::task::yield();
  auto view = rund::kernel::view<rund::kernel::i32>(
      storage->data(), rund::kernel::Index<1u>{storage->size()});
  *result = rund::kernel::each(
      rund::kernel::par(), rund::kernel::space(storage->size()),
      [&](auto index) noexcept {
        view(index) = static_cast<rund::kernel::i32>((index[0] * 17u) ^ 0x5au);
      });
}

} // namespace

int CheckRuntimeLifecycleStress();

int RunRuntimeStressContract() {
  std::uint64_t reference_task_hash = 0u;
  for (std::uint32_t run_index = 0u; run_index < 16u; ++run_index) {
    std::array<rund::kernel::i32, 96u> storage{};
    rund::kernel::SkeletonResult skeleton{};
    rund::task::Status joined{};
    const rund::Session::Result report = rund::run(
        rund::SessionConfig{
            .id = 80u,
            .workers = 3u,
            .scheduler =
                {
                    .task_capacity = 8u,
                    .ready_queue_capacity = 8u,
                },
        },
        [&] {
          const rund::task::Handle fill = rund::task::spawn(
              "nested-parallel", ProbeNestedParallel(&storage, &skeleton));
          joined = rund::task::join(fill);
        });
    TEST_ASSERT(report);
    TEST_ASSERT(joined);
    TEST_ASSERT(!skeleton.ok);
    TEST_ASSERT(skeleton.reason == std::string_view{"pool_nested_dispatch"});
    for (const rund::kernel::i32 value : storage) {
      TEST_ASSERT(value == 0);
    }
    if (run_index == 0u) {
      reference_task_hash = report.tasks().trace_hash();
    } else {
      TEST_ASSERT(report.tasks().trace_hash() == reference_task_hash);
    }
  }

  const rund::Session::Result failed_scope =
      rund::run(rund::SessionConfig{.workers = 2u}, [] { throw 7; });
  TEST_ASSERT(!failed_scope);
  TEST_ASSERT(failed_scope.error() ==
              std::string_view{"runtime_scope_callback_failed"});
  const rund::kernel::SkeletonResult missing_after_failure = rund::kernel::each(
      rund::kernel::par(), rund::kernel::space(8u), [](auto) noexcept {});
  TEST_ASSERT(!missing_after_failure.ok);
  TEST_ASSERT(missing_after_failure.reason ==
              std::string_view{"parallel_runtime_missing"});

  TEST_ASSERT(CheckRuntimeLifecycleStress() == 0);
  return 0;
}
