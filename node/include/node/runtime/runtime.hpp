#pragma once

#include <rund/task/status.hpp>
#include <rund/session.hpp>
#include <rund/telemetry/event.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rund::node {

struct ScopeEvidence;

namespace runtime_detail {
struct ComputeHostState;
struct RuntimeAccess;
}
} // namespace rund::node

namespace rund::replay::detail::scope {
struct Generation;
struct Lease;
struct Plan;
class Timing;
} // namespace rund::replay::detail::scope

namespace rund::compute::detail {
struct JobState;
} // namespace rund::compute::detail

namespace rund::node {

class Runtime {
public:
  ~Runtime();
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

private:
  Runtime();

  enum class Shutdown : std::uint8_t {
    Admission,
    Terminal,
  };

  [[nodiscard]] ::rund::Session::Status
  configure(::rund::SessionConfig options);
  [[nodiscard]] ::rund::Session::Status start();
  [[nodiscard]] ::rund::Session::Status shutdown(Shutdown intent);

  [[nodiscard]] ::rund::Session::Snapshot snapshot() const;
  [[nodiscard]] ::rund::Trace trace() const;
  [[nodiscard]] ::rund::Resources resources() const;

  using ScopeCallback = void (*)(void *);

  enum class TraceCapture : std::uint8_t {
    Scope,
    Deferred,
  };

  class TaskScopeFrame {
  public:
    TaskScopeFrame() noexcept = default;
    TaskScopeFrame(std::shared_ptr<void> lifetime, void *scheduler,
                   std::mutex &control,
                   const ::rund::replay::detail::scope::Plan &plan) noexcept;
    TaskScopeFrame(const TaskScopeFrame &) = delete;
    TaskScopeFrame &operator=(const TaskScopeFrame &) = delete;
    TaskScopeFrame(TaskScopeFrame &&) = delete;
    TaskScopeFrame &operator=(TaskScopeFrame &&) = delete;
    ~TaskScopeFrame();

    [[nodiscard]] explicit operator bool() const noexcept { return installed_; }
    [[nodiscard]] ReasonCode code() const noexcept { return code_; }
    [[nodiscard]] std::uint64_t id() const noexcept { return scope_id_; }
    [[nodiscard]] task::Status drain() noexcept;
    [[nodiscard]] ScopeEvidence evidence();

  private:
    void *previous_ = nullptr;
    void *scheduler_ = nullptr;
    std::shared_ptr<void> lifetime_{};
    std::unique_lock<std::mutex> control_{};
    std::uint64_t scope_id_ = 0u;
    std::uint64_t previous_scope_id_ = 0u;
    std::size_t observation_begin_ = 0u;
    std::size_t event_begin_ = 0u;
    bool installed_ = false;
    bool plan_installed_ = false;
    bool drained_ = false;
    ReasonCode code_ = ReasonCode::NodeRuntimeMissing;
  };

  struct State;
  struct ScopeAdmission final {
    ::rund::Session::Status status{};
    std::shared_ptr<runtime_detail::ComputeHostState> host{};

    [[nodiscard]] explicit operator bool() const noexcept {
      return static_cast<bool>(status);
    }
    [[nodiscard]] ReasonCode code() const noexcept { return status.code(); }
  };
  struct TraceCursor final {
    std::uint64_t next = 1u;
    std::uint64_t dropped = 0u;
  };

  [[nodiscard]] ::rund::Session::Result
  run_scope(const ::rund::replay::detail::scope::Plan &plan, void *callback,
            ScopeCallback invoke, TraceCapture trace_capture,
            ::rund::replay::detail::scope::Generation *generation,
            ::rund::replay::detail::scope::Lease *lease,
            ::rund::replay::detail::scope::Timing *timing);
  [[nodiscard]] static ::rund::Session::Result
  capture_result(TaskScopeFrame &scope, ReasonCode code, ::rund::Trace trace);
  [[nodiscard]] TraceCursor begin_trace() const noexcept;
  [[nodiscard]] ::rund::Trace capture_trace(TraceCursor begin) const;
  [[nodiscard]] TaskScopeFrame
  task_scope(const std::shared_ptr<runtime_detail::ComputeHostState> &host,
             const ::rund::replay::detail::scope::Plan &plan);
  [[nodiscard]] ScopeAdmission enter_scope();
  void leave_scope() noexcept;
  [[nodiscard]] ::rund::compute::Request
  compute_job(std::shared_ptr<compute::detail::JobState> job) noexcept;
  [[nodiscard]] ::rund::compute::Request compute_pipeline(
      std::shared_ptr<compute::detail::PipelineState> pipeline) noexcept;
  [[nodiscard]] ::rund::compute::Request
  compute_operation(std::shared_ptr<void> operation,
                    const void *operations) noexcept;
  void emit(::rund::telemetry::Event &&event) noexcept;

  std::unique_ptr<State> state_;

  friend class ::rund::Session;
  friend struct runtime_detail::RuntimeAccess;
};

} // namespace rund::node
