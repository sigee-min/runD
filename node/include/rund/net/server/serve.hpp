#pragma once

#include <rund/net/server/accept.hpp>
#include <rund/net/server/task.hpp>
#include <rund/task/group.hpp>
#include <rund/task/result.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::net::server {

static_assert(__atomic_always_lock_free(sizeof(std::uint32_t), nullptr));
static_assert(alignof(std::uint32_t) >= sizeof(std::uint32_t));

namespace detail {

inline constexpr std::uint64_t no_failure =
    std::numeric_limits<std::uint64_t>::max();

static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(
    std::is_same_v<std::underlying_type_t<::rund::ReasonCode>, std::uint16_t>);

struct Outcomes final {
  Result *result = nullptr;
  std::atomic<std::uint64_t> first_failure{no_failure};
};

enum class PeerTerminalClass : std::uint8_t {
  Completed,
  Stopped,
  Failed,
};

[[nodiscard]] inline constexpr PeerTerminalClass
classify_peer_terminal(const PeerResult terminal) noexcept {
  if (terminal) {
    return PeerTerminalClass::Completed;
  }
  return terminal.stopped() ? PeerTerminalClass::Stopped
                            : PeerTerminalClass::Failed;
}

[[nodiscard]] inline PeerResult handler_invocation_failure() noexcept {
  return PeerResult::fail(::rund::ReasonCode::NetPeerHandlerFailed);
}

[[nodiscard]] inline constexpr std::uint64_t
pack_failure(const std::uint32_t index,
             const ::rund::ReasonCode code) noexcept {
  return (static_cast<std::uint64_t>(index) << 16u) |
         static_cast<std::uint16_t>(code);
}

static_assert(pack_failure(std::numeric_limits<std::uint32_t>::max() - 1u,
                           static_cast<::rund::ReasonCode>(
                               std::numeric_limits<std::uint16_t>::max())) <
              no_failure);

[[nodiscard]] inline constexpr ::rund::ReasonCode
unpack_failure(const std::uint64_t value) noexcept {
  return static_cast<::rund::ReasonCode>(static_cast<std::uint16_t>(value));
}

inline void increment(std::uint32_t &value) noexcept {
  static_cast<void>(__atomic_fetch_add(&value, 1u, __ATOMIC_RELAXED));
}

inline void record(Outcomes &outcomes, const std::uint32_t index,
                   const PeerResult terminal) noexcept {
  switch (classify_peer_terminal(terminal)) {
  case PeerTerminalClass::Completed:
    increment(outcomes.result->completed);
    return;
  case PeerTerminalClass::Stopped:
    increment(outcomes.result->stopped);
    break;
  case PeerTerminalClass::Failed:
    increment(outcomes.result->failed);
    break;
  }

  const std::uint64_t candidate = pack_failure(index, terminal.code());
  std::uint64_t current =
      outcomes.first_failure.load(std::memory_order_relaxed);
  while (candidate < current &&
         !outcomes.first_failure.compare_exchange_weak(
             current, candidate, std::memory_order_release,
             std::memory_order_relaxed)) {
  }
}

[[nodiscard]] inline PeerResult
flatten(task::Result<PeerResult> result) noexcept {
  return result ? *std::move(result) : PeerResult::fail(result.code());
}

template <typename Handler>
[[nodiscard]] task::Task<void> run_peer(Handler *const handler, Peer peer,
                                        const std::uint32_t index,
                                        Outcomes *const outcomes) {
  try {
    record(*outcomes, index,
           flatten(co_await std::invoke(*handler, std::move(peer))));
  } catch (...) {
    record(*outcomes, index, handler_invocation_failure());
  }
}

[[nodiscard]] inline bool account_missing(Result &result) noexcept {
  const std::uint64_t terminal = static_cast<std::uint64_t>(result.completed) +
                                 result.failed + result.stopped;
  if (terminal < result.started) {
    result.failed += static_cast<std::uint32_t>(result.started - terminal);
    return true;
  }
  return false;
}

inline void finish(Result &result, const task::Status joined,
                   const Outcomes &outcomes) noexcept {
  const bool missing = account_missing(result);
  if (result.code() != ::rund::ReasonCode::Ok) {
    return;
  }
  if (!joined) {
    net::result::Access::set(result, joined.code());
    return;
  }
  const std::uint64_t first =
      outcomes.first_failure.load(std::memory_order_acquire);
  if (first != no_failure) {
    net::result::Access::set(result, unpack_failure(first));
  } else if (missing) {
    net::result::Access::set(result, ::rund::ReasonCode::NetPeerHandlerFailed);
  }
}

template <typename Handler>
[[nodiscard]] task::Task<Result> run(Options options, Handler handler) {
  static_assert(
      std::is_same_v<std::invoke_result_t<Handler &, Peer>,
                     task::Task<PeerResult>>,
      "serve handler must accept Peer and return task::Task<PeerResult>");

  Result result{::rund::ReasonCode::Ok};
  if (options.accepts.max_accepts == 0u) {
    result.budget_exhausted = true;
    co_return result;
  }

  ready::Ticket readable = co_await ready::read(options.listener);
  if (!readable) {
    net::result::Access::set(result, readable.code());
    co_return result;
  }

  for (std::uint32_t index = 0u; index < options.accepts.max_accepts; ++index) {
    accept::Result accepted = index == 0u ? accept::one(std::move(readable))
                                          : detail::next(options.listener);
    if (!accepted) {
      if (accepted.code() == ::rund::ReasonCode::IoWouldBlock) {
        result.would_block = true;
      } else {
        net::result::Access::set(result, accepted.code());
      }
      break;
    }

    detail::Prepared next =
        detail::prepare(std::move(accepted), options.accepted);
    if (!next) {
      net::result::Access::set(result, next.code());
      co_return result;
    }

    ++result.accepted;
    ++result.started;
    PeerResult handled{};
    try {
      handled = flatten(co_await std::invoke(handler, std::move(next.peer)));
    } catch (...) {
      handled = handler_invocation_failure();
    }
    switch (classify_peer_terminal(handled)) {
    case PeerTerminalClass::Completed:
      ++result.completed;
      break;
    case PeerTerminalClass::Stopped:
      ++result.stopped;
      net::result::Access::set(result, handled.code());
      co_return result;
    case PeerTerminalClass::Failed:
      ++result.failed;
      net::result::Access::set(result, handled.code());
      co_return result;
    }
  }
  result.budget_exhausted = result.accepted == options.accepts.max_accepts;
  co_return result;
}

template <typename Handler>
[[nodiscard]] task::Task<Result>
run(Options options, std::span<task::Handle> peer_tasks, Handler handler) {
  static_assert(
      std::is_same_v<std::invoke_result_t<Handler &, Peer>,
                     task::Task<PeerResult>>,
      "serve handler must accept Peer and return task::Task<PeerResult>");

  Result result{::rund::ReasonCode::Ok};
  if (options.accepts.max_accepts > peer_tasks.size()) {
    net::result::Access::set(result, ::rund::ReasonCode::TaskCapacityExceeded);
    co_return result;
  }

  task::Group tasks{peer_tasks};
  Outcomes outcomes{.result = &result};
  const char *const task_name =
      options.task_name != nullptr ? options.task_name : "net.peer";
  auto admit_peer = [&](accept::Result &&accepted) noexcept {
    detail::Prepared prepared =
        detail::prepare(std::move(accepted), options.accepted);
    if (!prepared) {
      net::result::Access::set(result, prepared.code());
      return false;
    }
    ++result.accepted;
    const std::uint32_t accept_index = result.accepted - 1u;
    const task::Handle peer_task = tasks.spawn(
        task_name, detail::run_peer(&handler, std::move(prepared.peer),
                                    accept_index, &outcomes));
    if (!peer_task) {
      ++result.rejected;
      net::result::Access::set(result, ::rund::ReasonCode::NetPeerSpawnFailed);
      return false;
    }
    ++result.started;
    return true;
  };

  const task::Result<accept::Drain> drained_task = co_await accept::drain(
      options.listener, options.accepts, std::move(admit_peer));
  if (!drained_task) {
    if (result.code() == ::rund::ReasonCode::Ok) {
      net::result::Access::set(result, ::rund::ReasonCode::NetAcceptTaskFailed);
    }
  } else {
    const accept::Drain &drained = *drained_task;
    result.would_block = drained.would_block;
    if (result.code() == ::rund::ReasonCode::Ok && !drained) {
      net::result::Access::set(result, drained.code());
    }
  }

  task::Group::JoinState joining = tasks.begin_join();
  while (joining.pending()) {
    joining.advance(co_await joining.current());
  }
  detail::finish(result, joining.finish(), outcomes);
  result.budget_exhausted = result.accepted == options.accepts.max_accepts;
  co_return result;
}

} // namespace detail

template <typename Handler>
[[nodiscard]] Task serve(Options options, Handler handler) {
  return Task{detail::run(options, std::move(handler))};
}

template <typename Handler>
[[nodiscard]] Task serve(Options options, std::span<task::Handle> peer_tasks,
                         Handler handler) {
  return Task{detail::run(options, peer_tasks, std::move(handler))};
}

} // namespace rund::net::server
