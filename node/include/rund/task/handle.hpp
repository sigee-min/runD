#pragma once

#include <rund/reason.hpp>

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace rund::task {

class SleepOp;
class Status;
class YieldOp;
class stop_source;
class stop_token;
template <typename T> class Result;
template <typename T> class Task;
template <typename T> class channel;
struct Handle;

} // namespace rund::task

namespace rund::host::chrono {
struct logical_clock;
} // namespace rund::host::chrono

namespace rund::host::random {
struct RunSeed;
RunSeed active_run_seed() noexcept;
} // namespace rund::host::random

namespace rund::host::env {
[[nodiscard]] task::Result<std::string> get(std::string_view name) noexcept;
} // namespace rund::host::env

namespace rund::detail::task {
class AwaitAccess;
class ChannelAccess;
class StopAccess;
class ApiAccess;
class HandleAccess;
} // namespace rund::detail::task

namespace rund::task {

[[nodiscard]] Status join_all(std::span<const Handle> handles) noexcept;
struct Handle {
public:
  constexpr Handle() noexcept = default;
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return id_ != 0u && code_ == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr std::uint64_t id() const noexcept { return id_; }
  [[nodiscard]] constexpr bool ok() const noexcept {
    return static_cast<bool>(*this);
  }
  [[nodiscard]] constexpr ReasonCode code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return code_ == ReasonCode::Ok ? std::string_view{}
                                   : std::string_view{ReasonString(code_)};
  }

private:
  friend class ::rund::detail::task::AwaitAccess;
  friend class ::rund::detail::task::ChannelAccess;
  friend class ::rund::detail::task::StopAccess;
  friend class ::rund::detail::task::ApiAccess;
  friend class ::rund::detail::task::HandleAccess;
  friend class stop_source;
  friend class stop_token;
  template <typename T> friend class channel;
  friend YieldOp yield() noexcept;
  friend struct host::chrono::logical_clock;
  friend host::random::RunSeed host::random::active_run_seed() noexcept;
  friend SleepOp sleep(std::chrono::nanoseconds duration) noexcept;
  friend Result<std::string> host::env::get(std::string_view name) noexcept;
  friend Status join(Handle handle) noexcept;
  friend Status join_all(std::span<const Handle> handles) noexcept;
  template <typename... Handles>
  friend Status join(Handle first, Handles... rest) noexcept;
  friend Handle spawn(const char *name, Task<void> &&task) noexcept;
  template <typename Callable>
  friend Handle spawn(const char *name, Callable &&callable);
  template <typename Callable> friend Status scope(Callable &&callable);

  constexpr Handle(const std::uint64_t id, const std::uint64_t scheduler_id,
                   const std::uint64_t scope_id, const ReasonCode code) noexcept
      : id_(id), scheduler_id_(scheduler_id), scope_id_(scope_id), code_(code) {
  }

  std::uint64_t id_ = 0u;
  std::uint64_t scheduler_id_ = 0u;
  std::uint64_t scope_id_ = 0u;
  ReasonCode code_ = ReasonCode::TaskHandleInvalid;
};

} // namespace rund::task

namespace rund::detail::task {

class HandleAccess final {
public:
  HandleAccess() = delete;

  [[nodiscard]] static constexpr ::rund::task::Handle
  Make(const std::uint64_t id, const std::uint64_t scheduler_id,
       const std::uint64_t scope_id, const ReasonCode code) noexcept {
    return ::rund::task::Handle{id, scheduler_id, scope_id, code};
  }

  [[nodiscard]] static constexpr ::rund::task::Handle
  Fail(const ReasonCode code) noexcept {
    return Make(0u, 0u, 0u, code);
  }

  [[nodiscard]] static constexpr std::uint64_t
  Scheduler(const ::rund::task::Handle &handle) noexcept {
    return handle.scheduler_id_;
  }

  [[nodiscard]] static constexpr std::uint64_t
  Scope(const ::rund::task::Handle &handle) noexcept {
    return handle.scope_id_;
  }
};

} // namespace rund::detail::task
