#include "model.hpp"

namespace rund::measure::scheduler {

rund::task::Task<void> YieldOnce(std::atomic<std::uint64_t> *const done) {
  const auto yielded = co_await rund::task::yield();
  if (yielded) {
    done->fetch_add(1u, std::memory_order_relaxed);
  }
}

rund::task::Task<void> JoinTree(const std::uint32_t depth,
                                std::atomic<std::uint64_t> *const leaves,
                                std::atomic_bool *const ok) {
  if (depth == 0u) {
    leaves->fetch_add(1u, std::memory_order_relaxed);
    co_return;
  }
  const auto left =
      rund::task::spawn("join-left", JoinTree(depth - 1u, leaves, ok));
  const auto right =
      rund::task::spawn("join-right", JoinTree(depth - 1u, leaves, ok));
  const auto left_joined = co_await left;
  const auto right_joined = co_await right;
  if (!left_joined || !right_joined) {
    ok->store(false, std::memory_order_relaxed);
  }
}

rund::task::Task<void> Ping(rund::task::channel<std::uint32_t> *const pipe,
                            const std::uint32_t rounds,
                            std::atomic_bool *const ok) {
  for (std::uint32_t index = 0u; index < rounds; ++index) {
    if (!(co_await pipe->send(index))) {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
    const auto value = co_await pipe->recv();
    if (!value || *value != index) {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
  }
}

rund::task::Task<void> Pong(rund::task::channel<std::uint32_t> *const pipe,
                            const std::uint32_t rounds,
                            std::atomic_bool *const ok) {
  for (std::uint32_t index = 0u; index < rounds; ++index) {
    const auto value = co_await pipe->recv();
    if (!value || *value != index || !(co_await pipe->send(index))) {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
  }
}

rund::task::Task<void> SleepLoop(const std::uint32_t rounds,
                                 std::atomic_bool *const ok) {
  for (std::uint32_t index = 0u; index < rounds; ++index) {
    if (!(co_await rund::task::sleep(std::chrono::microseconds{100}))) {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
  }
}

rund::task::Task<void> IoRead(Pipe *const data, Pipe *const ack,
                              const rund::host::io::FdView data_ready,
                              const std::uint32_t rounds,
                              std::atomic_bool *const ok) {
  for (std::uint32_t index = 0u; index < rounds; ++index) {
    const auto ready = co_await rund::host::io::readable(data_ready);
    char byte = 0;
    if (!ready || ::read(data->read_fd, &byte, 1u) != 1 || byte != 'd' ||
        ::write(ack->write_fd, "a", 1u) != 1) {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
  }
}

rund::task::Task<void> IoWrite(Pipe *const data, Pipe *const ack,
                               const rund::host::io::FdView ack_ready,
                               const std::uint32_t rounds,
                               std::atomic_bool *const ok) {
  for (std::uint32_t index = 0u; index < rounds; ++index) {
    if (!(co_await rund::task::yield()) ||
        ::write(data->write_fd, "d", 1u) != 1) {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
    const auto ready = co_await rund::host::io::readable(ack_ready);
    char byte = 0;
    if (!ready || ::read(ack->read_fd, &byte, 1u) != 1 || byte != 'a') {
      ok->store(false, std::memory_order_relaxed);
      co_return;
    }
  }
}

rund::task::Task<void> IoBatchWait(const rund::host::io::FdView fd,
                                   std::uint8_t *const ok) {
  const auto ready = co_await rund::host::io::readable(fd);
  if (!ready) {
    *ok = 0u;
  }
}

[[gnu::noinline]] rund::task::Task<void>
Hold(rund::task::channel<std::uint32_t> *const gate,
     std::atomic<std::uint32_t> *const started) {
  started->fetch_add(1u, std::memory_order_release);
  (void)co_await gate->recv();
}

[[gnu::noinline]] rund::task::Task<void>
HoldGroup(rund::task::channel<std::uint32_t> *const gate,
          const std::span<rund::task::Handle> handles,
          std::atomic<std::uint32_t> *const started,
          std::uint64_t *const active_rss, std::uint64_t *const active_heap,
          bool *const ok, std::string_view *const reason) {
  for (std::size_t index = 0u; index < handles.size(); ++index) {
    handles[index] = rund::task::spawn("held", Hold(gate, started));
    if (!handles[index]) {
      *ok = false;
      *reason = handles[index].error();
      co_return;
    }
  }
  while (started->load(std::memory_order_acquire) != handles.size()) {
    if (!(co_await rund::task::yield())) {
      *ok = false;
      *reason = "task_start_yield_failed";
      co_return;
    }
  }
  *active_rss = RssBytes();
  *active_heap = HeapBytes();
  const auto closed = gate->close();
  if (!closed) {
    *ok = false;
    *reason = closed.error();
    co_return;
  }
}


} // namespace rund::measure::scheduler
