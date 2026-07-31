#include "model.hpp"

namespace rund::measure::scheduler {

[[nodiscard]] int Latency() {
  bool all_ok = true;
  constexpr std::uint32_t batch = 4096u;
  std::vector<rund::task::Handle> handles{};
  handles.reserve(batch);

  const Measure spawn = RunMeasure(Config(batch + 16u), batch, [&] {
    handles.clear();
    for (std::uint32_t index = 0u; index < batch; ++index) {
      handles.push_back(rund::task::spawn("leaf", [] {}));
    }
    return Joined(handles);
  });
  Print("spawn_complete", spawn);
  all_ok = all_ok && spawn.ok;

  std::atomic<std::uint64_t> yielded{0u};
  const Measure yield = RunMeasure(Config(batch + 16u), batch, [&] {
    const std::uint64_t before = yielded.load(std::memory_order_relaxed);
    handles.clear();
    for (std::uint32_t index = 0u; index < batch; ++index) {
      handles.push_back(rund::task::spawn("yield", YieldOnce(&yielded)));
    }
    return Joined(handles) &&
           yielded.load(std::memory_order_relaxed) - before == batch;
  });
  Print("yield", yield);
  all_ok = all_ok && yield.ok;

  constexpr std::uint32_t tree_depth = 10u;
  constexpr std::uint64_t tree_nodes = (1ull << (tree_depth + 1u)) - 1u;
  std::atomic<std::uint64_t> leaves{0u};
  std::atomic_bool tree_ok{true};
  const Measure join_tree = RunMeasure(
      Config(static_cast<std::uint32_t>(tree_nodes + 16u)), tree_nodes, [&] {
        const std::uint64_t before = leaves.load(std::memory_order_relaxed);
        tree_ok.store(true, std::memory_order_relaxed);
        const auto root = rund::task::spawn(
            "join-root", JoinTree(tree_depth, &leaves, &tree_ok));
        return static_cast<bool>(rund::task::join(root)) &&
               tree_ok.load(std::memory_order_relaxed) &&
               leaves.load(std::memory_order_relaxed) - before ==
                   (1ull << tree_depth);
      });
  Print("join_tree", join_tree);
  all_ok = all_ok && join_tree.ok;

  constexpr std::uint32_t exchanges = 4096u;
  const Measure channel = RunMeasure(Config(16u), exchanges, [&] {
    auto pipe = rund::task::channel<std::uint32_t>::make(0u);
    std::atomic_bool ok{true};
    const auto ping = rund::task::spawn("ping", Ping(&pipe, exchanges, &ok));
    const auto pong = rund::task::spawn("pong", Pong(&pipe, exchanges, &ok));
    const bool joined = static_cast<bool>(rund::task::join(ping, pong));
    (void)pipe.close();
    return joined && ok.load(std::memory_order_relaxed);
  });
  Print("channel_ping_pong", channel);
  all_ok = all_ok && channel.ok;

  constexpr std::uint32_t timer_rounds = 64u;
  const Measure timer = RunMeasure(Config(16u), timer_rounds, [&] {
    std::atomic_bool ok{true};
    const auto task = rund::task::spawn("timer", SleepLoop(timer_rounds, &ok));
    return static_cast<bool>(rund::task::join(task)) &&
           ok.load(std::memory_order_relaxed);
  });
  Print("timer_park_wake", timer);
  all_ok = all_ok && timer.ok;

  constexpr std::uint32_t io_rounds = 1024u;
  Pipe data{};
  Pipe ack{};
  if (!data || !ack) {
    std::fprintf(stderr, "pipe creation failed\n");
    return 1;
  }
  rund::host::io::Fd data_ready =
      rund::host::io::take_native_fd(::dup(data.read_fd));
  rund::host::io::Fd ack_ready =
      rund::host::io::take_native_fd(::dup(ack.read_fd));
  if (!data_ready || !ack_ready) {
    std::fprintf(stderr, "pipe admission failed\n");
    return 1;
  }
  const Measure io = RunMeasure(Config(16u), io_rounds, [&] {
    std::atomic_bool ok{true};
    const auto reader = rund::task::spawn(
        "io-read", IoRead(&data, &ack, data_ready.view(), io_rounds, &ok));
    const auto writer = rund::task::spawn(
        "io-write", IoWrite(&data, &ack, ack_ready.view(), io_rounds, &ok));
    return static_cast<bool>(rund::task::join(reader, writer)) &&
           ok.load(std::memory_order_relaxed);
  });
  Print("io_park_wake", io);
  all_ok = all_ok && io.ok;

  constexpr std::size_t io_batch_waits = 1024u;
  constexpr std::size_t io_batch_fds = 64u;
  std::array<Pipe, io_batch_fds> batch_pipes{};
  for (const Pipe &pipe : batch_pipes) {
    if (!pipe) {
      std::fprintf(stderr, "batch pipe creation failed\n");
      return 1;
    }
  }
  std::array<rund::host::io::Fd, io_batch_fds> batch_ready_fds{};
  for (std::size_t index = 0u; index < io_batch_fds; ++index) {
    batch_ready_fds[index] =
        rund::host::io::take_native_fd(::dup(batch_pipes[index].read_fd));
    if (!batch_ready_fds[index]) {
      std::fprintf(stderr, "batch pipe admission failed\n");
      return 1;
    }
  }
  std::vector<std::uint8_t> batch_ready(io_batch_waits, 1u);
  rund::SessionConfig batch_config = Config(io_batch_waits + 32u);
  batch_config.scheduler.observation_capacity =
      static_cast<std::uint32_t>((kWarmRounds + 1u) * io_batch_waits);
  batch_config.scheduler.host_event_capacity =
      static_cast<std::uint32_t>((kWarmRounds + 1u) * io_batch_waits);
  const Measure io_batch = RunMeasure(batch_config, io_batch_waits, [&] {
    std::fill(batch_ready.begin(), batch_ready.end(), 1u);
    handles.clear();
    for (std::size_t index = 0u; index < io_batch_waits; ++index) {
      handles.push_back(rund::task::spawn(
          "io-batch-wait",
          IoBatchWait(batch_ready_fds[index % io_batch_fds].view(),
                      &batch_ready[index])));
    }
    bool round_ok = true;
    handles.push_back(rund::task::spawn("io-batch-write", [&] {
      for (Pipe &pipe : batch_pipes) {
        if (::write(pipe.write_fd, "b", 1u) != 1) {
          round_ok = false;
        }
      }
    }));
    round_ok = Joined(handles) && round_ok;
    for (Pipe &pipe : batch_pipes) {
      char byte = 0;
      if (::read(pipe.read_fd, &byte, 1u) != 1 || byte != 'b') {
        round_ok = false;
      }
    }
    return round_ok &&
           std::all_of(batch_ready.begin(), batch_ready.end(),
                       [](const std::uint8_t ready) { return ready != 0u; });
  });
  Print("reactor_ready_batch", io_batch);
  all_ok = all_ok && io_batch.ok;
  return all_ok ? 0 : 1;
}


} // namespace rund::measure::scheduler
