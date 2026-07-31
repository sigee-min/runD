#include "model.hpp"

namespace rund::measure::scheduler {

[[nodiscard]] std::uint64_t RssBytes() noexcept {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (::task_info(::mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
    return static_cast<std::uint64_t>(info.resident_size);
  }
#endif
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0u;
  }
#if defined(__APPLE__)
  return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024u;
#endif
}

[[nodiscard]] std::uint64_t HeapBytes() noexcept {
#if defined(__APPLE__)
  malloc_statistics_t stats{};
  malloc_zone_statistics(malloc_default_zone(), &stats);
  return static_cast<std::uint64_t>(stats.size_in_use);
#else
  return 0u;
#endif
}


[[nodiscard]] rund::SessionConfig
Config(const std::uint32_t capacity, const std::uint32_t task_workers) {
  return rund::SessionConfig{
      .workers = 1u,
      .trace_capacity = 1u,
      .scheduler =
          {
              .task_workers = task_workers,
              .task_capacity = capacity,
              .ready_queue_capacity = capacity,
              .coroutine_frame_bytes = 512u,
              .task_result_bytes = 64u,
              .timer_capacity = capacity,
              .channel_capacity = 4u,
              .channel_buffer_capacity = 4u,
              .channel_wait_capacity = capacity,
              .reactor_wait_capacity = capacity,
              .observation_capacity = 1u,
              .host_event_capacity = 1u,
              .host_payload_capacity_bytes = 1u,
          },
  };
}

[[nodiscard]] bool Joined(const std::span<const rund::task::Handle> tasks) {
  return static_cast<bool>(rund::task::join_all(tasks));
}


} // namespace rund::measure::scheduler
