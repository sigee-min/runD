#pragma once

#include "../../pipeline/transfer/batch.hpp"
#include "../active.hpp"
#include "../state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct VirtualRunProjection final {
  VirtualActiveProjection active{};
  std::uint64_t input_capacity_bytes{};
  std::uint64_t input_visible_bytes{};
  std::uint64_t cache_extent{};
  std::uint64_t cache_identity_hi{};
  std::uint64_t cache_identity_lo{};
  std::uint64_t input_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t input_payload_bytes{};
  std::uint64_t output_payload_bytes{};
  std::uint64_t input_prefix_bytes{};
  std::uint64_t output_prefix_bytes{};
  std::uint64_t input_frame_elements{};
  bool clamp_window{};
  bool clip_window{};
  bool reduction{};
  bool scan{};
  bool inclusive_scan{};
  Type input_type{Type::I32};
  Type output_type{Type::I32};
  std::uint32_t operation{};
  std::uint64_t frame_capacity{};
  std::uint64_t input_arena_bytes{};
  std::uint64_t output_arena_bytes{};
  std::byte *input_stage{};
  std::byte *output_stage{};
  PipelineFrameUpload upload{};
  PipelineFrameDownload download{};
};

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

[[nodiscard]] bool
project_virtual_run(VirtualPipelineState &state, std::uint64_t active_count,
                    VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
bind_virtual_run_transfer(VirtualPipelineState &state,
                          VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_epoch(const VirtualRunProjection &run, std::uint64_t epoch,
                      VirtualEpochProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_input_page(const VirtualRunProjection &run, std::uint64_t page,
                           VirtualInputPageProjection &projection) noexcept;

} // namespace rund::compute::detail
