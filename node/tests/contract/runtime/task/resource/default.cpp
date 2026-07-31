#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/session.hpp>

#include <cstddef>

int RunRuntimeTaskDefaultResourceBudgetContract() {
  const rund::SessionConfig spec{};
  const rund::SchedulerConfig &scheduler = spec.scheduler;
  TEST_ASSERT(scheduler.task_workers == 0u);
  TEST_ASSERT(scheduler.task_capacity == 1024u);
  TEST_ASSERT(scheduler.ready_queue_capacity == 1024u);
  TEST_ASSERT(scheduler.coroutine_frame_bytes == 4096u);
  TEST_ASSERT(scheduler.coroutine_frame_alignment == 16u);
  TEST_ASSERT(scheduler.task_result_bytes == 128u);
  TEST_ASSERT(scheduler.task_result_alignment == 64u);
  TEST_ASSERT(scheduler.timer_capacity == 1024u);
  TEST_ASSERT(scheduler.channel_capacity == 1024u);
  TEST_ASSERT(scheduler.channel_buffer_capacity == 65536u);
  TEST_ASSERT(scheduler.channel_wait_capacity == 4096u);
  TEST_ASSERT(scheduler.reactor_wait_capacity == 1024u);
  TEST_ASSERT(scheduler.net_ready_set_capacity == 256u);
  TEST_ASSERT(scheduler.net_ready_set_member_capacity == 4096u);
  TEST_ASSERT(scheduler.net_iov_capacity == 64u);
  TEST_ASSERT(scheduler.net_datagram_capacity_bytes == 65507u);
  TEST_ASSERT(scheduler.net_socket_registry_capacity == 65536u);
  TEST_ASSERT(scheduler.host_handle_capacity == 1024u);
  TEST_ASSERT(scheduler.reactor_ready_budget == 0u);
  TEST_ASSERT(scheduler.observation_capacity == 0u);
  TEST_ASSERT(scheduler.host_io_capacity == 64u);
  TEST_ASSERT(scheduler.host_event_capacity == 1024u);
  TEST_ASSERT(scheduler.host_payload_capacity_bytes == 1048576u);

  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(rund::SessionConfig{}, [&] {
    const rund::task::Handle task =
        rund::task::spawn("default-resource-budget", [] {});
    joined = rund::task::join(task);
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(report.tasks().task_workers() >= 1u);
  TEST_ASSERT(report.tasks().resources().max_tasks() == 1024u);
  TEST_ASSERT(report.tasks().resources().max_reactor_waits() == 1024u);
  TEST_ASSERT(report.tasks().resources().max_ready_sets() == 256u);
  TEST_ASSERT(report.tasks().resources().max_ready_set_members() == 4096u);
  TEST_ASSERT(report.tasks().resources().max_socket_registry_entries() == 65536u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_capacity() == 1024u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_bytes() == 4096u);
  TEST_ASSERT(report.tasks().resources().coroutine_frames_live() == 0u);
  TEST_ASSERT(report.tasks().resources().coroutine_frame_failures() == 0u);

  return 0;
}
