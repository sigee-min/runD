#pragma once

#include "../../backend/transfer/model.hpp"
#include "../../device/residency/prefetch.hpp"
#include "../../device/residency/registry.hpp"
#include "projection.hpp"

#include <rund/compute/stats.hpp>

#include <array>
#include <cstddef>
#include <span>

namespace rund::compute {
class VirtualBacking;
}

namespace rund::compute::detail {
struct PipelineState;
struct VirtualRunTransaction;

// Returns a complete backend-authenticated writable view only when it is the
// exact private Input owner for this Direct bank. The matching Authority lease
// remains the only mutation permission.
[[nodiscard]] BufferWriteView
residency_input_view(PipelineState &pipeline,
                     const VirtualRunProjection &run) noexcept;

// Returns a complete authenticated read-only view only for the exact private
// Direct Output owner. Authority still owns the writeback permission.
[[nodiscard]] BufferReadView
residency_output_view(PipelineState &pipeline) noexcept;

struct VirtualTransferInterval final {
  std::uint64_t started_ns{};
  std::uint64_t completed_ns{};
};

struct VirtualOutputReservation final {
  std::array<residency::CacheBinding, PipelineLeafCapacity> bindings{};
  std::array<residency::CacheTransition, PipelineLeafCapacity * 3u>
      transitions{};
  std::size_t binding_count{};
  std::size_t transition_count{};
  std::uint64_t token{};
};

[[nodiscard]] bool project_residency_transform_uses(
    const VirtualEpochProjection &epoch, const VirtualRunProjection &run,
    std::span<residency::CacheUse> input_uses,
    std::span<residency::CacheUse> output_uses) noexcept;

[[nodiscard]] Status reserve_residency_output(
    PipelineState &pipeline, const VirtualEpochProjection &epoch,
    const VirtualRunProjection &run, VirtualOutputReservation &reservation,
    bool &poison) noexcept;

[[nodiscard]] residency::EpochLease
residency_output_lease(const PipelineState &pipeline,
                       residency::EpochLease execution,
                       VirtualOutputReservation &reservation) noexcept;

[[nodiscard]] bool
cancel_residency_output(PipelineState &pipeline,
                        VirtualOutputReservation &reservation) noexcept;

[[nodiscard]] Status
supply_residency_cache(PipelineState &pipeline, const VirtualRunProjection &run,
                       residency::EpochLease lease, Stats &stats,
                       const residency::PrefetchReceipt *host = nullptr,
                       VirtualTransferInterval *interval = nullptr) noexcept;

[[nodiscard]] Status retain_residency_output(PipelineState &pipeline,
                                             const VirtualRunProjection &run,
                                             residency::EpochLease lease,
                                             Stats &stats) noexcept;

[[nodiscard]] Status retain_residency_output(
    PipelineState &pipeline, const VirtualRunProjection &run,
    const residency::ExecutionTicket &ticket, Stats &stats) noexcept;

// Focused contract hook. The next retention boundary fails before Authority
// publication so exact rollback and same-owner retry can be proved.
void inject_residency_retention_failure_once() noexcept;

// Focused contract cleanup. Returns true only when the legacy retention
// boundary was not selected and the one-shot injection remained pending.
[[nodiscard]] bool cancel_residency_retention_failure_injection() noexcept;

[[nodiscard]] Status writeback_residency_cache(
    VirtualBacking &output, const VirtualRunProjection &run,
    std::span<const residency::CacheTransition> transitions,
    ResidencyStats &residency_stats,
    VirtualTransferInterval *interval = nullptr,
    std::span<const std::byte *const> frames = {},
    VirtualRunTransaction *transaction = nullptr) noexcept;

// Writes exact Authority-authenticated output extents without publishing the
// backing generation. A whole-run execution stages every output first, closes
// Authority, and then publishes once. Existing epoch execution uses the
// combined writeback_residency_cache convenience above.
[[nodiscard]] Status
stage_residency_cache(VirtualBacking &output, const VirtualRunProjection &run,
                      std::span<const residency::CacheTransition> transitions,
                      ResidencyStats &residency_stats,
                      VirtualTransferInterval *interval = nullptr,
                      std::span<const std::byte *const> frames = {},
                      VirtualRunTransaction *transaction = nullptr) noexcept;

void publish_residency_cache(
    VirtualBacking &output,
    VirtualRunTransaction *transaction = nullptr) noexcept;

} // namespace rund::compute::detail
