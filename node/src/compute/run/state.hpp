#pragma once

#include <rund/compute/abi/model.hpp>
#include <rund/compute/stats.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

struct RunState final {
  std::shared_ptr<ProgramState> program;
  std::array<std::shared_ptr<BufferState>, MaxOutputs> outputs{};
  mutable Stats stats{};
  // Program execution remains successful when a numeric primitive publishes
  // a per-batch semantic status. Pipeline alone consumes this retained first
  // semantic failure and projects it to its terminal publication law.
  Primitive semantic_primitive{Primitive::Reduce};
  std::uint32_t semantic_status{};
  std::uint64_t semantic_failure_count{};
  mutable std::array<std::uint64_t, MaxOutputs> output_hashes{};
  // Executable Compute shapes are u32-addressable. Retaining the same width
  // avoids widening every output receipt to the host pointer width.
  mutable std::array<std::uint32_t, MaxOutputs> output_hash_counts{};
  mutable std::uint32_t output_hash_mask{};
};

} // namespace rund::compute::detail
