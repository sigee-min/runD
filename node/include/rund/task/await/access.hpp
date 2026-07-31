#pragma once

#include <rund/task/handle.hpp>
#include <rund/task/results.hpp>

namespace rund::task {

class HandleAwaiter;
class IoAwaiter;
class Group;
class SleepAwaiter;
class YieldAwaiter;
template <typename T> class TaskAwaiter;

} // namespace rund::task

namespace rund::net::ready::many {
class Awaiter;
class Wait;
struct Result;
} // namespace rund::net::ready::many

namespace rund::net::ready {
class Awaiter;
class Ticket;
} // namespace rund::net::ready

namespace rund::net {
class SocketView;
}

namespace rund::net::ready::timed {
class Awaiter;
class Wait;
} // namespace rund::net::ready::timed

namespace rund::detail::task {

class AwaitAccess final {
private:
  AwaitAccess() = delete;

  friend class rund::task::HandleAwaiter;
  friend class rund::task::IoAwaiter;
  friend class rund::task::Group;
  friend class rund::task::SleepAwaiter;
  friend class rund::task::YieldAwaiter;
  template <typename T> friend class rund::task::TaskAwaiter;
  friend class rund::net::ready::timed::Awaiter;
  friend class rund::net::ready::many::Awaiter;
  friend class rund::net::ready::Awaiter;

  [[nodiscard]] static ::rund::detail::task::AwaitDecision
  BeginCoroutineJoinAwait(const ::rund::task::Handle &handle) noexcept;
  [[nodiscard]] static ::rund::task::Status CompleteCoroutineJoinAwait(
      ::rund::detail::task::AwaitDecision decision) noexcept;
  static void RetireCoroutineTask(const ::rund::task::Handle &handle) noexcept;
  [[nodiscard]] static ::rund::detail::task::AwaitDecision
  SuspendCoroutineYield() noexcept;
  [[nodiscard]] static ::rund::detail::task::AwaitDecision
  SuspendCoroutineSleep(std::chrono::nanoseconds duration) noexcept;
  [[nodiscard]] static ::rund::task::IoResult
  CompleteCoroutineIoAwait(::rund::detail::task::IoDecision decision) noexcept;
  [[nodiscard]] static ::rund::detail::task::IoDecision
  SuspendCoroutineIo(int fd, short interest, std::uint64_t host_handle_id,
                     std::uint64_t fd_generation) noexcept;
  [[nodiscard]] static ::rund::detail::task::IoDecision
  SuspendCoroutineNetIo(::rund::net::SocketView socket,
                        short interest) noexcept;
  [[nodiscard]] static ::rund::task::IoResult
  CompleteCoroutineNetIo(::rund::detail::task::IoDecision decision) noexcept;
  [[nodiscard]] static ::rund::net::ready::Ticket
  CompleteCoroutineTimedReadyAwait(
      ::rund::net::ready::timed::Wait operation) noexcept;
  [[nodiscard]] static ::rund::net::ready::timed::Wait
  SuspendCoroutineTimedReady(
      ::rund::net::ready::timed::Wait operation) noexcept;
  [[nodiscard]] static ::rund::net::ready::many::Wait
  SuspendCoroutineReadyMany(::rund::net::ready::many::Wait operation) noexcept;
  [[nodiscard]] static ::rund::net::ready::many::Result
  CompleteCoroutineReadyMany(::rund::net::ready::many::Wait operation) noexcept;
};

} // namespace rund::detail::task
