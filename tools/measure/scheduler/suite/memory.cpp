#include "model.hpp"

namespace rund::measure::scheduler {

[[nodiscard]] int Memory(const std::uint32_t tasks,
                         const std::uint32_t task_workers) {
  if (tasks == 0u || tasks > 100000u) {
    std::fprintf(stderr, "memory task count must be 1..100000\n");
    return 2;
  }
  std::uint32_t observed_frame_bytes = 0u;
  bool probe_ok = true;
  const rund::Session::Result probe = rund::run(Config(16u), [&] {
    auto gate = rund::task::channel<std::uint32_t>::make(0u);
    std::atomic<std::uint32_t> started{0u};
    auto task = Hold(&gate, &started);
    if (task) {
      observed_frame_bytes = rund::detail::task::frame::Bytes(
          rund::detail::task::CoroutineAddress(task));
    }
    std::array<rund::task::Handle, 1u> slots{};
    std::uint64_t rss = 0u;
    std::uint64_t heap = 0u;
    bool group_ok = true;
    std::string_view reason = "ok";
    auto group =
        HoldGroup(&gate, slots, &started, &rss, &heap, &group_ok, &reason);
    if (group) {
      observed_frame_bytes =
          std::max(observed_frame_bytes,
                   rund::detail::task::frame::Bytes(
                       rund::detail::task::CoroutineAddress(group)));
    }
    auto discarded = rund::detail::task::TakeCoroutine(std::move(group));
    rund::detail::task::DestroyCoroutine(discarded);
    const auto handle = rund::task::spawn("frame-probe", std::move(task));
    probe_ok = static_cast<bool>(handle) && static_cast<bool>(gate.close()) &&
               static_cast<bool>(rund::task::join(handle));
  });
  if (!probe || !probe_ok || observed_frame_bytes == 0u) {
    const std::string_view error = probe.error();
    std::fprintf(stderr, "coroutine frame probe failed: %.*s\n",
                 static_cast<int>(error.size()), error.data());
    return 1;
  }

  std::uint64_t configured = 0u;
  std::uint64_t heap_configured = 0u;
  std::uint64_t active = 0u;
  std::uint64_t heap_active = 0u;
  std::atomic<std::uint32_t> started{0u};
  bool round_ok = true;
  std::string_view round_reason = "ok";
  std::vector<rund::task::Handle> handles(tasks);
  rund::SessionConfig config = Config(tasks + 16u, task_workers);
  config.scheduler.coroutine_frame_bytes = 4096u;
  config.scheduler.task_result_bytes = 128u;
  config.scheduler.timer_capacity = 1u;
  config.scheduler.channel_wait_capacity = tasks + 16u;
  config.scheduler.reactor_wait_capacity = 1u;
  const std::uint64_t before = RssBytes();
  const std::uint64_t heap_before = HeapBytes();
  const rund::Session::Result report = rund::run(config, [&] {
    configured = RssBytes();
    heap_configured = HeapBytes();
    auto gate = rund::task::channel<std::uint32_t>::make(0u);
    const auto coordinator = rund::task::spawn(
        "hold-group", HoldGroup(&gate, handles, &started, &active, &heap_active,
                                &round_ok, &round_reason));
    const auto joined = rund::task::join(coordinator);
    if (!joined) {
      round_ok = false;
      round_reason = joined.error();
    }
    const auto children = rund::task::join_all(handles);
    if (!children) {
      round_ok = false;
      round_reason = children.error();
    }
  });
  const auto resources = report.tasks().resources();
  const std::uint64_t frame_limit =
      resources.coroutine_frame_capacity() * resources.coroutine_frame_bytes();
  const std::string_view report_reason =
      report ? (round_ok ? std::string_view{"ok"} : round_reason)
             : report.error();
  std::printf(
      "task_memory ok=%u reason=%.*s tasks=%u workers=%u rss_before=%llu "
      "rss_configured=%llu rss_active=%llu rss_configured_delta=%llu "
      "rss_live_delta=%llu rss_delta=%llu heap_before=%llu "
      "heap_configured=%llu heap_active=%llu heap_configured_delta=%llu "
      "heap_live_delta=%llu heap_delta=%llu "
      "frame_capacity=%llu frame_limit_bytes=%llu "
      "observed_frame_bytes=%u frame_high_water=%llu frame_alloc=%llu "
      "frame_reuse=%llu "
      "frame_fail=%llu\n",
      report && round_ok ? 1u : 0u, static_cast<int>(report_reason.size()),
      report_reason.data(), tasks, task_workers,
      static_cast<unsigned long long>(before),
      static_cast<unsigned long long>(configured),
      static_cast<unsigned long long>(active),
      static_cast<unsigned long long>(configured > before ? configured - before
                                                          : 0u),
      static_cast<unsigned long long>(active > configured ? active - configured
                                                          : 0u),
      static_cast<unsigned long long>(active > before ? active - before : 0u),
      static_cast<unsigned long long>(heap_before),
      static_cast<unsigned long long>(heap_configured),
      static_cast<unsigned long long>(heap_active),
      static_cast<unsigned long long>(
          heap_configured > heap_before ? heap_configured - heap_before : 0u),
      static_cast<unsigned long long>(
          heap_active > heap_configured ? heap_active - heap_configured : 0u),
      static_cast<unsigned long long>(
          heap_active > heap_before ? heap_active - heap_before : 0u),
      static_cast<unsigned long long>(resources.coroutine_frame_capacity()),
      static_cast<unsigned long long>(frame_limit), observed_frame_bytes,
      static_cast<unsigned long long>(resources.coroutine_frames_high_water()),
      static_cast<unsigned long long>(resources.coroutine_frame_allocations()),
      static_cast<unsigned long long>(resources.coroutine_frame_reuses()),
      static_cast<unsigned long long>(resources.coroutine_frame_failures()));
  return report && round_ok ? 0 : 1;
}


} // namespace rund::measure::scheduler
