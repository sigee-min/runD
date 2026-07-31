#pragma once

#include <cstdint>

namespace rund {

struct SchedulerConfig {
  std::uint32_t task_workers = 0u;
  std::uint32_t task_capacity = 1024u;
  std::uint32_t ready_queue_capacity = 1024u;
  std::uint32_t coroutine_frame_bytes = 4096u;
  std::uint32_t coroutine_frame_alignment = 16u;
  std::uint32_t task_result_bytes = 128u;
  std::uint32_t task_result_alignment = 64u;
  std::uint32_t timer_capacity = 1024u;
  std::uint32_t channel_capacity = 1024u;
  std::uint32_t channel_buffer_capacity = 65536u;
  std::uint32_t channel_wait_capacity = 4096u;
  std::uint32_t reactor_wait_capacity = 1024u;
  std::uint32_t net_ready_set_capacity = 256u;
  std::uint32_t net_ready_set_member_capacity = 4096u;
  std::uint32_t net_iov_capacity = 64u;
  std::uint32_t net_datagram_capacity_bytes = 65507u;
  std::uint32_t net_socket_registry_capacity = 65536u;
  std::uint32_t reactor_ready_budget = 0u;
  std::uint32_t observation_capacity = 0u;
  std::uint32_t host_handle_capacity = 1024u;
  std::uint32_t host_io_capacity = 64u;
  std::uint32_t host_event_capacity = 1024u;
  std::uint64_t host_payload_capacity_bytes = 1048576u;
};

} // namespace rund
