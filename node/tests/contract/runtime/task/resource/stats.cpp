#include "test/assert.hpp"

#include <rund/net/limits.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>

int RunRuntimeTaskResourceStatsContract() {
  const rund::SessionConfig config{
      .id = 989u,
      .workers = 1u,
      .trace_capacity = 64u,
      .scheduler =
          {
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
              .reactor_wait_capacity = 8u,
              .net_ready_set_capacity = 4u,
              .net_ready_set_member_capacity = 16u,
              .net_iov_capacity = 7u,
              .net_socket_registry_capacity = 32u,
              .host_event_capacity = 64u,
          },
  };

  rund::net::ready::Status created{};
  rund::net::ready::Status destroyed{};
  rund::net::Limits limits{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(config, [&] {
    const rund::task::Handle task =
        rund::task::spawn("runtime-resource-stats", [&] {
          created = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 4u});
          if (created.ok()) {
            limits = rund::net::limits();
            destroyed = rund::net::ready::destroy(created.set);
          }
        });
    joined = rund::task::join(task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(created.ok());
  TEST_ASSERT(destroyed.ok());
  TEST_ASSERT(limits.ok());
  TEST_ASSERT(limits.max_ready_sets == 4u);
  TEST_ASSERT(limits.max_ready_set_members == 16u);
  TEST_ASSERT(limits.max_iov == 7u);
  TEST_ASSERT(limits.max_socket_registry_entries == 32u);
  TEST_ASSERT(report.tasks().task_workers() == 1u);
  TEST_ASSERT(report.tasks().resources().max_tasks() == 8u);
  TEST_ASSERT(report.tasks().resources().max_reactor_waits() == 8u);
  TEST_ASSERT(report.tasks().resources().max_ready_sets() == 4u);
  TEST_ASSERT(report.tasks().resources().max_ready_set_members() == 16u);
  TEST_ASSERT(report.tasks().resources().max_socket_registry_entries() == 32u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_capacity() == 8u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_bytes() == 4096u);
  TEST_ASSERT(report.tasks().resources().coroutine_frames_live() == 0u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_allocations() == 0u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_failures() == 0u);

  return 0;
}
