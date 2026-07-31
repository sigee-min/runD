#pragma once

#include "state.hpp"

#include <rund/reason.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::runtime_detail {

void RetireSubmitted(const std::shared_ptr<ComputeHostState> &host) noexcept;
void CancelSubmitted(const std::shared_ptr<ComputeHostState> &host) noexcept;

} // namespace rund::node::runtime_detail

namespace rund::node {

void Signal(runtime_detail::ComputeHostState *host, ::rund::TraceEvent event,
            compute::Reason reason = compute::Reason::Ok) noexcept;
[[nodiscard]] bool
InSchedulerTask(const runtime_detail::ComputeHostState &host) noexcept;
[[nodiscard]] bool
OwnsSchedulerControl(const runtime_detail::ComputeHostState &host) noexcept;
[[nodiscard]] compute::Reason SpawnReason(::rund::ReasonCode reason) noexcept;

[[nodiscard]] compute::Result<compute::Backend>
OperationBackend(const compute_detail::Operation &operation) noexcept;
[[nodiscard]] kernel::u32
OperationWorkers(const compute_detail::Operation &operation) noexcept;
[[nodiscard]] compute::Status
ReserveOperation(const compute_detail::Operation &operation) noexcept;
[[nodiscard]] compute::Stats
OperationEvidence(const compute_detail::Operation &operation) noexcept;
void RecordOperationFrame(const compute_detail::Operation &operation,
                          std::uint64_t bytes, bool reused,
                          std::uint64_t budget) noexcept;
void ReleaseOperationFrame(const compute_detail::Operation &operation,
                           std::uint64_t bytes) noexcept;
[[nodiscard]] compute::Status
FinishOperation(compute_detail::TaskState &task,
                compute::Status failure) noexcept;
void Complete(compute_detail::TaskState *task, const compute::Status &status,
              const compute::Stats &stats = {}) noexcept;

[[nodiscard]] task::Task<void>
RunCpuCoordinator(compute_detail::TaskState *task);
[[nodiscard]] task::Task<void>
RunAccelCoordinator(compute_detail::TaskState *task);

[[nodiscard]] compute_detail::TaskState *
ClaimTask(runtime_detail::ComputeHostState &host,
          const compute_detail::Operation &operation);
void Retire(const std::shared_ptr<runtime_detail::ComputeHostState> &host,
            compute_detail::TaskState *task) noexcept;
void Release(compute_detail::TaskState *&task) noexcept;

} // namespace rund::node
