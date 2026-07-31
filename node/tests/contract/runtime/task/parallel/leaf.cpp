#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/program/executor.hpp>
#include <kernel/program/skeleton.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <string_view>

int RunRuntimeLeafParallelContract() {
  std::array<rund::kernel::i32, 64u> storage{};
  rund::kernel::SkeletonResult nested{};
  rund::task::Status children{};
  const rund::Session::Result report =
      rund::run(rund::SessionConfig{
        .id = 79u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 8u,
          .ready_queue_capacity = 8u,
        },
      },
                [&] {
                  children = rund::task::scope([&] {
                    (void)rund::task::spawn("leaf-parallel", [&] {
                      auto view = rund::kernel::view<rund::kernel::i32>(
                          storage.data(),
                          rund::kernel::Index<1u>{storage.size()});
                      nested = rund::kernel::each(
                          rund::kernel::par(),
                          rund::kernel::space(storage.size()),
                          [&](auto index) noexcept {
                            view(index) = static_cast<rund::kernel::i32>(
                                index[0] + 11u);
                          });
                    });
                  });
                });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(children.ok());
  TEST_ASSERT(!nested.ok);
  TEST_ASSERT(nested.reason == std::string_view{"pool_nested_dispatch"});
  for (const rund::kernel::i32 value : storage) {
    TEST_ASSERT(value == 0);
  }

  constexpr std::size_t kLeafTasks = 64u;
  std::array<rund::task::Status, kLeafTasks> scope_status{};
  std::array<rund::task::Handle, kLeafTasks> handles{};
  std::atomic<std::uint64_t> scope_callbacks{0u};
  rund::task::Status joined{};
  const rund::Session::Result scope_report = rund::run(
      rund::SessionConfig{
          .id = 80u,
          .workers = 1u,
          .scheduler = {
              .task_workers = 4u,
              .task_capacity = kLeafTasks,
              .ready_queue_capacity = kLeafTasks,
          },
      },
      [&] {
        for (std::size_t index = 0u; index < handles.size(); ++index) {
          handles[index] = rund::task::spawn("leaf-scope-rejected", [&, index] {
            scope_status[index] = rund::task::scope([&] {
              scope_callbacks.fetch_add(1u, std::memory_order_relaxed);
            });
          });
          TEST_ASSERT(handles[index]);
        }
        joined = rund::task::join_all(handles);
      });
  TEST_ASSERT(scope_report);
  TEST_ASSERT(!joined);
  TEST_ASSERT(joined.code() ==
              rund::ReasonCode::TaskLeafPrimitiveForbidden);
  TEST_ASSERT(scope_callbacks.load(std::memory_order_relaxed) == 0u);
  TEST_ASSERT(scope_report.tasks().failed() == kLeafTasks);
  TEST_ASSERT(scope_report.tasks().completed() == 0u);
  for (const rund::task::Status status : scope_status) {
    TEST_ASSERT(!status);
    TEST_ASSERT(status.code() ==
                rund::ReasonCode::TaskLeafPrimitiveForbidden);
  }

  return 0;
}
