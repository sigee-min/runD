#pragma once

#include "admission.hpp"
#include "publication.hpp"

#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund::compute::detail {

namespace residency {
class VirtualTransactionOwner;
} // namespace residency

enum class VirtualRunWriteCertainty : std::uint8_t {
  KnownNoWrite,
  UnknownMayWrite,
};

struct VirtualRunTransaction final {
  static constexpr std::size_t OutputRegionCount = 2u;
  static constexpr std::size_t JournalCapacity =
      2u * PipelineLeafCapacity + 2u * PipelineIterationCapacity;
  using Journal =
      std::array<residency::VirtualTransactionLease::Row, JournalCapacity>;

  VirtualBacking *output{};
  VirtualBackingTransaction *provider{};
  // Scan transactions authenticate every Authority operation against the
  // exact pool owner captured at admission.  This is a credential, not a
  // replaceable state lookup.
  residency::Authority *authority{};
  VirtualBackingTransactionToken token{};
  VirtualRunPublicationCursor cursor{};
  // Captured from the sealed run projection before provider begin. The device
  // and Host banks are the only rows a product transaction may inspect; no
  // logical-page journal or Pool reconstruction is permitted here.
  std::array<residency::FrameRegion, OutputRegionCount> output_regions{};
  std::array<residency::FrameRegion, OutputRegionCount> host_output_regions{};
  residency::VirtualTransactionLease output_lease{};
  std::uint64_t result_identity_hi{};
  std::uint64_t result_identity_lo{};
  std::uint64_t cache_extent{};
  std::uint64_t boundary_page{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t page_count{};
  std::uint64_t host_output_frame_capacity{};
  std::unique_ptr<Journal> journal{};
  std::size_t journal_count{};
  bool scan{};
  bool rows_active{};
  bool started{};
};

struct VirtualPipelineState;

[[nodiscard]] Status record_transaction_output(
    VirtualRunTransaction &, const residency::VirtualTransactionOwner &,
    std::uint64_t lease_token,
    std::uint32_t frame, residency::CacheKey, residency::FrameTier) noexcept;

// A BackendUnsupported provider is a capability fallback and therefore
// returns success with an inactive transaction. All other begin failures are
// terminal before any execution owner is created.
[[nodiscard]] Status begin_virtual_run_transaction(
    VirtualPipelineState &, const VirtualRunAdmission &,
    const VirtualRunProjection &, VirtualBacking &, std::uint64_t page_count,
    VirtualRunTransaction &) noexcept;

[[nodiscard]] Status prepare_virtual_run_transaction(
    VirtualPipelineState &, VirtualRunTransaction &) noexcept;
// Captures the fixed physical output rows after provider/Pipeline preflight.
// Success requires the exact expected row count; abort capture permits zero.
[[nodiscard]] Status prepare_virtual_run_transaction_rows(
    VirtualPipelineState &, VirtualRunTransaction &, bool require_rows) noexcept;
[[nodiscard]] Status commit_virtual_run_transaction(
    VirtualPipelineState &, VirtualRunTransaction &, bool &poison_pipeline,
    VirtualRunWriteCertainty &effective_certainty) noexcept;
void abort_virtual_run_transaction(VirtualPipelineState &,
                                   VirtualRunTransaction &,
                                   VirtualRunWriteCertainty,
                                   VirtualRunWriteCertainty &effective_certainty,
                                   Status &status, bool &poison_pipeline) noexcept;

} // namespace rund::compute::detail
