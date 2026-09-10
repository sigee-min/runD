#pragma once

#include "../../device/residency/registry.hpp"
#include "../../pipeline/transfer/batch.hpp"
#include "../active.hpp"
#include "../state.hpp"

#include <kernel/program/compute/scan/model.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct VirtualRunInputProjection final {
  std::uint64_t capacity_bytes{};
  std::uint64_t visible_bytes{};
  std::uint64_t identity_hi{};
  std::uint64_t identity_lo{};
  std::uint64_t backing{};
  std::uint64_t version{};
  Type type{Type::I32};
};

// Exclusive logical topology. Derived queries never store independent flags.
enum class VirtualRunTopology : std::uint8_t {
  Direct,
  LocalWindow,
  Reduction,
  Scan,
  MultiPointwise,
  MultiScan,
  GraphPointwise,
  GraphReduction,
};

struct VirtualRunProjection final {
  VirtualActiveProjection active{};
  // Canonical Program-port order. DeviceVsm consumes this fixed set directly;
  // the scalar fields below remain the legacy unary/Graph projection surface.
  std::array<VirtualRunInputProjection, VirtualPipelineState::InputCapacity>
      inputs{};
  std::size_t input_count{};
  std::uint64_t input_capacity_bytes{};
  std::uint64_t input_visible_bytes{};
  std::uint64_t cache_extent{};
  // Backing input identity depends only on typed physical framing. Result and
  // transient identities retain the canonical Graph fingerprint. Keeping the
  // two domains separate permits truthful cross-Graph/cache-class reuse.
  std::uint64_t input_identity_hi{};
  std::uint64_t input_identity_lo{};
  // Canonical backing-page identity excludes halo expansion and boundary
  // semantics. Centered Window uses it for the distinct fixed-size
  // FootprintProjection while retaining `input_identity_*` for expanded-frame
  // cache compatibility during migration.
  std::uint64_t canonical_input_identity_hi{};
  std::uint64_t canonical_input_identity_lo{};
  std::uint64_t result_identity_hi{};
  std::uint64_t result_identity_lo{};
  // Backing identity is frozen once under both backing locks. Every epoch in
  // this invocation binds to the same generations even when page-out
  // publishes intermediate backing batches.
  std::uint64_t input_backing{};
  std::uint64_t input_version{};
  std::uint64_t output_backing{};
  std::uint64_t output_version{};
  std::uint64_t input_page_bytes{};
  std::uint64_t intermediate_page_bytes{};
  std::uint64_t control_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t input_payload_bytes{};
  std::uint64_t output_payload_bytes{};
  std::uint64_t input_prefix_bytes{};
  std::uint64_t output_prefix_bytes{};
  std::uint64_t input_frame_elements{};
  bool clamp_window{};
  bool clip_window{};
  bool device_vsm_required{};
  VirtualRunTopology topology{VirtualRunTopology::Direct};
  Type input_type{Type::I32};
  Type output_type{Type::I32};
  std::uint32_t operation{};
  std::uint64_t frame_capacity{};
  std::uint64_t input_arena_bytes{};
  std::uint64_t intermediate_arena_bytes{};
  std::uint64_t control_arena_bytes{};
  std::uint64_t output_arena_bytes{};
  std::array<residency::FrameRegion, 2u> input_regions{};
  std::array<residency::FrameRegion, 2u> intermediate_regions{};
  std::array<residency::FrameRegion, 2u> output_regions{};
  // Direct streams still own contiguous first/count regions. Graph execution
  // must use the exact arrays above because a borrowed arena's bank stride C
  // may be larger than this Pool's issued capacity K.
  std::uint32_t first_intermediate_frame{};
  std::uint32_t first_output_frame{};
  std::uint32_t first_host_input_frame{};
  std::uint32_t host_input_count{};
  std::uint32_t host_frame_capacity{};
  std::uint32_t first_host_output_frame{};
  std::uint32_t host_output_frame_capacity{};
  // CPU execution aliases these authoritative Host frame owners directly.
  // Accelerators use two VSM-registered Host banks consumed directly by H2D;
  // no prefetch payload or staging mirror may sit between these pointers and
  // the execution bank.
  std::array<std::byte *, 2u> input_host_banks{};
  std::array<std::byte *, 2u> control_host_banks{};
  std::array<std::byte *, 2u> output_host_banks{};
  residency::GraphMaterialization graph_input{};
  residency::GraphMaterialization graph_intermediate{};
  residency::GraphMaterialization graph_output{};
  std::array<std::uint32_t, residency::TiledGraphResourceCapacity>
      graph_resources{};
  std::array<residency::GraphMaterialization,
             residency::TiledGraphResourceCapacity>
      graph_materializations{};
  std::size_t graph_resource_count{};
  PipelineFrameUpload upload{};
  PipelineFrameDownload download{};

  [[nodiscard]] constexpr bool graph_execution() const noexcept {
    return topology == VirtualRunTopology::GraphPointwise || graph_reduction();
  }
  [[nodiscard]] constexpr bool graph_reduction() const noexcept {
    return topology == VirtualRunTopology::GraphReduction;
  }
  [[nodiscard]] constexpr bool reduction() const noexcept {
    return topology == VirtualRunTopology::Reduction || graph_reduction();
  }
  [[nodiscard]] constexpr bool multi_pointwise() const noexcept {
    return topology == VirtualRunTopology::MultiPointwise;
  }
  [[nodiscard]] constexpr bool multi_scan() const noexcept {
    return topology == VirtualRunTopology::MultiScan;
  }
  [[nodiscard]] constexpr bool scan() const noexcept {
    return topology == VirtualRunTopology::Scan || multi_scan();
  }
  [[nodiscard]] constexpr bool inclusive_scan() const noexcept {
    return scan() && operation == static_cast<std::uint32_t>(
                                      kernel::ScanOp::InclusiveSum);
  }
  [[nodiscard]] constexpr bool poolless_device_vsm() const noexcept {
    return multi_pointwise() || multi_scan();
  }
};

[[nodiscard]] std::byte *virtual_input_frame(const VirtualRunProjection &run,
                                             std::uint32_t frame) noexcept;

[[nodiscard]] std::byte *virtual_output_frame(const VirtualRunProjection &run,
                                              std::uint32_t frame) noexcept;

[[nodiscard]] std::byte *
virtual_host_input_frame(const VirtualRunProjection &run,
                         std::uint32_t frame) noexcept;

[[nodiscard]] residency::FrameRegion
virtual_graph_host_input_region(const VirtualRunProjection &run,
                                std::size_t input, std::uint32_t bank) noexcept;

[[nodiscard]] std::byte *
virtual_host_output_frame(const VirtualRunProjection &run,
                          std::uint32_t frame) noexcept;

[[nodiscard]] std::byte *
virtual_resident_output_frame(const VirtualRunProjection &run,
                              std::uint32_t frame) noexcept;

struct VirtualEpochProjection final {
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t page_count{};
  std::uint64_t input_offset{};
  std::uint64_t output_offset{};
  std::size_t logical_input_bytes{};
  std::size_t logical_output_bytes{};
};

struct VirtualInputPageProjection final {
  std::uint64_t logical_offset{};
  std::size_t target_offset{};
  std::size_t transfer_bytes{};
  std::size_t leading_fill_bytes{};
  std::size_t trailing_fill_offset{};
};

// Exact overlap between two adjacent materialized input footprints. The copy
// slice and the remaining backing read are both expressed in bytes so backing,
// prefetch, and whole-run Host service consume one projection authority.
struct VirtualInputReuseProjection final {
  std::size_t source_offset{};
  std::size_t target_offset{};
  std::size_t bytes{};
  std::uint64_t logical_offset{};
  std::size_t read_target_offset{};
  std::size_t read_bytes{};
};

[[nodiscard]] bool
project_virtual_run(VirtualPipelineState &state, std::uint64_t active_count,
                    VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
bind_virtual_run_transfer(VirtualPipelineState &state,
                          VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
bind_virtual_run_transfer(PipelineState &pipeline,
                          VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_epoch(const VirtualRunProjection &run, std::uint64_t epoch,
                      VirtualEpochProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_input_page(const VirtualRunProjection &run, std::uint64_t page,
                           VirtualInputPageProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_input_reuse(const VirtualInputPageProjection &prior,
                            const VirtualInputPageProjection &current,
                            VirtualInputReuseProjection &projection) noexcept;

// Projects the frozen plan's page-local dirty slice into its logical output
// address space. Backed outputs use the backing byte range; transient
// collective partials use the VSM materialization range and are never passed
// to backing writeback.
[[nodiscard]] bool
project_virtual_output_dirty(const VirtualRunProjection &run,
                             std::uint64_t page,
                             residency::DirtyExtent &projection) noexcept;

[[nodiscard]] residency::CacheKey
virtual_cache_key(const VirtualRunProjection &run, std::uint64_t backing,
                  std::uint64_t version, std::uint64_t page) noexcept;

[[nodiscard]] residency::CacheKey
virtual_output_cache_key(const VirtualRunProjection &run, std::uint64_t backing,
                         std::uint64_t version, std::uint64_t page) noexcept;

} // namespace rund::compute::detail
