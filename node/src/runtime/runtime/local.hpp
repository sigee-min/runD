#pragma once

#include <node/resource/envelope.hpp>
#include <node/runtime/runtime.hpp>

#include <kernel/program/executor.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rund::node::runtime_detail {

struct ComputeHostState;

[[nodiscard]] bool RuntimeActive(const Runtime *runtime);

struct RuntimeActiveScope {
  explicit RuntimeActiveScope(const Runtime *runtime);
  ~RuntimeActiveScope();

  const Runtime *runtime = nullptr;
  RuntimeActiveScope *previous = nullptr;
};

[[nodiscard]] bool Runnable(::rund::SessionState state);
[[nodiscard]] bool
HostReplayStorageValid(const ::rund::replay::Storage &config) noexcept;
[[nodiscard]] bool
HostReplayDiagnosticValid(const ::rund::replay::Diagnostic &config) noexcept;
[[nodiscard]] ::rund::TraceRecord
MakeTrace(::rund::TraceEvent event, ::rund::TraceCode code,
          const ::rund::Session::Snapshot &snapshot, std::uint64_t sequence);
[[nodiscard]] ::rund::Session::Status LifecycleFail(ReasonCode code,
                                                    ::rund::SessionState state);
[[nodiscard]] ::rund::Session::Status LifecyclePass(::rund::SessionState state);
[[nodiscard]] ::rund::SessionState ObserveLifecycle(const Runtime &runtime);
[[nodiscard]] kernel::ParallelRuntime
AcquireParallelRuntime(void *context, kernel::u32 workers);
[[nodiscard]] kernel::executor_detail::ScopedParallelRuntimeProvider
InstallKernelScope(Runtime &runtime);

struct RuntimeAccess final {
  [[nodiscard]] static Runtime::State &state(Runtime &runtime) noexcept {
    return *runtime.state_;
  }

  [[nodiscard]] static const Runtime::State &
  state(const Runtime &runtime) noexcept {
    return *runtime.state_;
  }

  [[nodiscard]] static ::rund::Trace take_trace(Runtime &runtime);
};

} // namespace rund::node::runtime_detail

namespace rund::node {

static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

struct Runtime::State {
  mutable std::mutex mutex{};
  std::condition_variable scope_drained{};
  std::uint64_t id = 0u;
  ResourceEnvelope resources{};
  ::rund::telemetry::Sink telemetry{};
  ::rund::SessionState lifecycle = ::rund::SessionState::Unconfigured;
  std::size_t trace_capacity = 1024u;
  std::size_t trace_head = 0u;
  std::uint64_t next_trace = 1u;
  std::uint64_t dropped_trace = 0u;
  std::vector<::rund::TraceRecord> trace{};
  bool scope_active = false;
  std::atomic<std::uint64_t> active_scope{0u};
  std::shared_ptr<runtime_detail::ComputeHostState> compute_host{};

  [[nodiscard]] ::rund::Session::Snapshot SnapshotLocked() const;

  void AddTrace(const ::rund::TraceEvent event, const ::rund::TraceCode code) {
    if (trace_capacity == 0u) {
      ++dropped_trace;
      return;
    }
    ::rund::TraceRecord record =
        runtime_detail::MakeTrace(event, code, SnapshotLocked(), next_trace++);
    if (trace.size() < trace_capacity) {
      trace.push_back(record);
      return;
    }
    trace[trace_head] = record;
    ++trace_head;
    if (trace_head == trace.size()) {
      trace_head = 0u;
    }
    ++dropped_trace;
  }

  ::rund::Trace TraceLocked() const {
    ::rund::Trace snapshot{.dropped = dropped_trace};
    snapshot.records.reserve(trace.size());
    std::size_t index = trace_head;
    for (std::size_t copied = 0u; copied < trace.size(); ++copied) {
      snapshot.records.push_back(trace[index]);
      ++index;
      if (index == trace.size()) {
        index = 0u;
      }
    }
    return snapshot;
  }

  ::rund::Trace TakeTraceLocked() {
    ::rund::Trace snapshot{.dropped = dropped_trace};
    if (trace_head != 0u) {
      std::rotate(trace.begin(), trace.begin() + trace_head, trace.end());
    }
    snapshot.records = std::move(trace);
    trace_head = 0u;
    return snapshot;
  }
};

} // namespace rund::node
