#pragma once

#include "../../pipeline/transfer/batch.hpp"
#include "../active.hpp"
#include "../state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct VirtualRunProjection final {
  VirtualActiveProjection active{};
  std::uint64_t input_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t slot_capacity{};
  std::uint64_t input_arena_bytes{};
  std::uint64_t output_arena_bytes{};
  std::byte *input_stage{};
  std::byte *output_stage{};
  PipelineSlotUpload upload{};
  PipelineSlotDownload download{};
};

struct VirtualWaveProjection final {
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t page_count{};
  std::uint64_t input_offset{};
  std::uint64_t output_offset{};
  std::size_t logical_input_bytes{};
  std::size_t logical_output_bytes{};
};

[[nodiscard]] bool
project_virtual_run(VirtualPipelineState &state, std::uint64_t active_count,
                    VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
bind_virtual_run_transfer(VirtualPipelineState &state,
                          VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_wave(const VirtualRunProjection &run, std::uint64_t wave,
                     VirtualWaveProjection &projection) noexcept;

} // namespace rund::compute::detail
