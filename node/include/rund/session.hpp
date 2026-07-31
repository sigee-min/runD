#pragma once

#include <rund/host/event.hpp>
#include <rund/session/config.hpp>
#include <rund/session/memory.hpp>
#include <rund/session/resources.hpp>
#include <rund/session/state.hpp>
#include <rund/session/trace.hpp>
#include <rund/task/observation.hpp>
#include <rund/task/stats.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rund {

namespace detail::run {
struct Access;
}

namespace replay::detail {
struct Access;
}

namespace compute {
template <class> class Job;
class Request;
class Pipeline;
namespace detail {
struct JobState;
struct PipelineState;
struct SessionDeviceAccess;
} // namespace detail
} // namespace compute

namespace replay::detail::scope {
struct Access;
struct Lease;
struct Plan;
class Timing;
} // namespace replay::detail::scope

namespace detail::session {
struct ResultAccess;
struct StatusAccess;
} // namespace detail::session

class Session final {
public:
  struct Status final {
    constexpr Status() noexcept = default;

    [[nodiscard]] constexpr bool ok() const noexcept {
      return code_ == ::rund::ReasonCode::Ok;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
      return ok();
    }
    [[nodiscard]] std::string_view error() const noexcept {
      return ok() ? std::string_view{}
                  : std::string_view{::rund::ReasonString(code_)};
    }
    [[nodiscard]] constexpr int exit_code() const noexcept {
      return ok() ? 0 : 1;
    }
    [[nodiscard]] constexpr ::rund::ReasonCode code() const noexcept {
      return code_;
    }
    [[nodiscard]] constexpr SessionState state() const noexcept {
      return state_;
    }

  private:
    constexpr Status(const ::rund::ReasonCode code,
                     const SessionState state) noexcept
        : code_(code), state_(state) {}

    ::rund::ReasonCode code_ = ::rund::ReasonCode::NotConfigured;
    SessionState state_ = SessionState::Unconfigured;

    friend struct detail::session::StatusAccess;
  };

  class Result final {
  public:
    Result() noexcept;
    ~Result();
    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;
    Result(Result &&other) noexcept;
    Result &operator=(Result &&other) noexcept;

    [[nodiscard]] constexpr bool ok() const noexcept {
      return code_ == ::rund::ReasonCode::Ok;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
      return ok();
    }
    [[nodiscard]] std::string_view error() const noexcept {
      return ok() ? std::string_view{}
                  : std::string_view{::rund::ReasonString(code_)};
    }
    [[nodiscard]] constexpr int exit_code() const noexcept {
      return ok() ? 0 : 1;
    }
    [[nodiscard]] constexpr ::rund::ReasonCode code() const noexcept {
      return code_;
    }
    [[nodiscard]] constexpr std::uint64_t scope() const noexcept {
      return scope_;
    }
    [[nodiscard]] constexpr const task::Stats &tasks() const noexcept {
      return tasks_;
    }
    [[nodiscard]] constexpr const PreparedMemory &memory() const noexcept {
      return memory_;
    }
    [[nodiscard]] std::span<const task::Observation>
    observations() const noexcept {
      return observations_;
    }
    [[nodiscard]] std::span<const ::rund::host::Event> events() const noexcept {
      return events_;
    }
    [[nodiscard]] constexpr const Trace &trace() const noexcept {
      return trace_;
    }
    [[nodiscard]] std::uint64_t trace_hash() const noexcept;

  private:
    Result(::rund::ReasonCode code, std::uint64_t scope, task::Stats tasks,
           PreparedMemory memory, std::vector<task::Observation> observations,
           std::vector<::rund::host::Event> events,
           const std::uint64_t input_rows, const std::uint64_t input_bytes,
           const std::uint64_t ready_capacity, Trace trace,
           telemetry::Detail detail = {},
           std::uint8_t telemetry_clock_reads = 0u) noexcept;

    [[nodiscard]] void *storage() noexcept;

    // Stable inline ABI budget for private replay evidence. The implementation
    // proves its archive fits this slot; the public header does not mirror the
    // archive's standard-library-dependent layout.
    static constexpr std::size_t payload_capacity = 320u;

    ::rund::ReasonCode code_ = ::rund::ReasonCode::SessionResultMissing;
    std::uint8_t telemetry_clock_reads_ = 0u;
    std::uint64_t scope_ = 0u;
    task::Stats tasks_{};
    PreparedMemory memory_{};
    std::vector<task::Observation> observations_{};
    std::vector<::rund::host::Event> events_{};
    alignas(std::max_align_t) std::byte payload_[payload_capacity]{};
    std::uint64_t input_rows_ = 0u;
    std::uint64_t input_bytes_ = 0u;
    std::uint64_t ready_capacity_ = 0u;
    Trace trace_{};
    telemetry::Detail detail_{};

    friend struct detail::session::ResultAccess;
  };

  struct Snapshot final {
    SessionState state = SessionState::Unconfigured;
    std::uint32_t active_compute_jobs = 0u;
    bool scope_active = false;
    ::rund::ReasonCode code = ::rund::ReasonCode::Ok;

    [[nodiscard]] constexpr bool ok() const noexcept {
      return code == ::rund::ReasonCode::Ok;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
      return ok();
    }
    [[nodiscard]] std::string_view error() const noexcept {
      return ok() ? std::string_view{}
                  : std::string_view{::rund::ReasonString(code)};
    }
    [[nodiscard]] constexpr int exit_code() const noexcept {
      return ok() ? 0 : 1;
    }
  };

  Session();
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  [[nodiscard]] Status open(SessionConfig config);
  [[nodiscard]] Status drain();
  [[nodiscard]] Status close();

  template <typename Callback> [[nodiscard]] Result scope(Callback &&callback) {
    static_assert(std::invocable<Callback &>,
                  "Session::scope callback must accept no arguments");
    CallbackRef<Callback> callback_ref{callback};
    return run_scope(&callback_ref, invoke_scope<Callback>);
  }

  template <class Signature>
  [[nodiscard]] compute::Request compute(compute::Job<Signature> &job) noexcept;
  [[nodiscard]] compute::Request compute(compute::Pipeline &pipeline) noexcept;

  [[nodiscard]] Snapshot snapshot() const;
  [[nodiscard]] Trace trace() const;
  [[nodiscard]] Resources resources() const;

private:
  using ScopeCallback = void (*)(void *);

  template <typename Callback> struct CallbackRef final {
    Callback &value;
  };

  template <typename Callback> static void invoke_scope(void *const raw) {
    auto &callback = static_cast<CallbackRef<Callback> *>(raw)->value;
    callback();
  }

  struct State;
  [[nodiscard]] Result run_scope(void *callback, ScopeCallback invoke);
  [[nodiscard]] Result
  run_scope(const ::rund::replay::detail::scope::Plan &plan, void *callback,
            ScopeCallback invoke, ::rund::replay::detail::scope::Lease *lease,
            ::rund::replay::detail::scope::Timing *timing);
  [[nodiscard]] Result terminal(void *callback, ScopeCallback invoke);
  [[nodiscard]] Trace take_trace();
  [[nodiscard]] compute::Request
  compute_job(std::shared_ptr<compute::detail::JobState> job) noexcept;
  [[nodiscard]] compute::Request
  compute_pipeline(std::shared_ptr<compute::detail::PipelineState> pipeline)
      noexcept;
  void emit(telemetry::Event &&event) noexcept;

  std::unique_ptr<State> state_;

  friend struct detail::run::Access;
  friend struct replay::detail::Access;
  friend struct compute::detail::SessionDeviceAccess;
  friend struct ::rund::replay::detail::scope::Access;
};

} // namespace rund

#include <rund/session/run.hpp>
