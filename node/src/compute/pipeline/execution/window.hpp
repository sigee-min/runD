#pragma once

#include "attempt.hpp"
#include "../../device/residency/execution/window.hpp"

#include "../../../accel/kernel/prepared/interface/residency.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rund::compute::detail {

struct PipelineWindowRelease final {
  residency::execution::Release release{};
  std::uint64_t attempt_generation{};
  std::uint64_t published_generation{};
  bool terminal_published{};
  bool reseeded{};
};

// Fixed Compute-side terminal journal for W4. It does not submit commands and
// cannot replace the common Pipeline finish/publish/reseed owner.
class PipelineExecutionWindow final {
public:
  [[nodiscard]] bool bind(const residency::execution::Plan &,
                          const residency::ExecutionLease &) noexcept;
  [[nodiscard]] bool bind(const residency::execution::Plan &,
                          const residency::ExecutionLease &,
                          std::uint64_t first_epoch,
                          std::size_t count) noexcept;
  [[nodiscard]] bool terminal(const PipelineWindowRelease &) noexcept;
  [[nodiscard]] bool final(std::uint64_t queue_calls,
                           std::uint64_t native_inflight_peak,
                           std::uint64_t completed_ns,
                           residency::execution::WindowEvidence &) noexcept;
  [[nodiscard]] bool integrity(std::uint64_t queue_calls,
                               std::uint64_t native_inflight_peak,
                               std::uint64_t completed_ns,
                               residency::execution::WindowEvidence &) const
      noexcept;
  void reset() noexcept;

private:
  const residency::execution::Plan *plan_{};
  residency::ExecutionLease lease_{};
  std::array<std::uint64_t, residency::execution::WindowCapacity> slot_epoch_{};
  std::array<PipelineWindowRelease, residency::execution::WindowCapacity>
      releases_{};
  std::uint64_t first_epoch_{};
  std::size_t expected_count_{};
  std::size_t count_{};
  bool failed_{};
  bool unknown_{};
};

struct PipelineState;

using PipelineResidencyWindowReleaseCompletion =
    void (*)(void *, PipelineWindowRelease &&) noexcept;
using PipelineResidencyWindowFinalCompletion =
    void (*)(void *, residency::execution::WindowEvidence &&) noexcept;

// Caller-owned fixed W<=4 Compute terminal owner. The raw prepared layer owns
// native commands only; this object is the sole bridge through start,
// control/evidence validation, terminal publication, and recurrent reseed.
// Its lifetime must extend through the one final callback.
struct PipelineResidencyWindowControl final {
  std::mutex gate{};
  node::accel::detail::PreparedResidencyWindowControl native{};
  PipelineExecutionWindow evidence{};
  const residency::execution::Plan *plan{};
  residency::ExecutionLease lease{};
  std::array<std::shared_ptr<PipelineState>,
             residency::execution::WindowCapacity>
      pipelines{};
  std::array<PipelineExecutionAttempt,
             residency::execution::WindowCapacity>
      attempts{};
  PipelineResidencyWindowReleaseCompletion release{};
  PipelineResidencyWindowFinalCompletion final{};
  void *user{};
  bool active{};
  bool unknown_seen{};
};

[[nodiscard]] Status submit_pipeline_execution_window(
    const residency::execution::Plan &, const residency::ExecutionLease &,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineResidencyWindowReleaseCompletion,
    PipelineResidencyWindowFinalCompletion, void *,
    PipelineResidencyWindowControl &) noexcept;

[[nodiscard]] Status submit_pipeline_execution_window_chunk(
    const residency::execution::Plan &, const residency::ExecutionLease &,
    std::uint64_t first_epoch, std::size_t count,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineResidencyWindowReleaseCompletion,
    PipelineResidencyWindowFinalCompletion, void *,
    PipelineResidencyWindowControl &) noexcept;

[[nodiscard]] Status submit_pipeline_execution_stream_chunk(
    const residency::execution::Plan &, const residency::ExecutionLease &,
    std::uint64_t first_epoch, std::size_t count,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineResidencyWindowReleaseCompletion,
    PipelineResidencyWindowFinalCompletion, void *,
    PipelineResidencyWindowControl &,
    node::accel::detail::PreparedResidencyStreamControl &) noexcept;

// `admission` is the exact Host-service result for this epoch. Success starts
// and prepares the Compute Pipeline before opening its native gate. Failure
// opens the already queued backend batch only through its no-write suppress
// gate and creates no Pipeline attempt.
[[nodiscard]] Status
signal_pipeline_execution_window(PipelineResidencyWindowControl &,
                                 std::uint64_t epoch,
                                 Status admission) noexcept;

// Emergency terminal for an already accepted native window when an exact
// ready/suppress signal cannot be delivered. It never reports success: the
// backend opens every remaining gate as UnknownMayWrite and quarantines the
// complete prepared owner before producing the one Final callback.
[[nodiscard]] Status
abort_pipeline_execution_window(PipelineResidencyWindowControl &,
                                Status failure) noexcept;

// Rearms fixed chunk-local storage after the raw/common Final has released its
// bounded claims and before a recurrent controller submits the next chunk.
// This never releases or reacquires the controller's future whole-stream
// Pipeline claim.
[[nodiscard]] Status
rearm_pipeline_execution_window(PipelineResidencyWindowControl &) noexcept;

} // namespace rund::compute::detail
