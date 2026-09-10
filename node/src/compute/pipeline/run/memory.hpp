#pragma once

#include "../../../accel/kernel/memory.hpp"
#include "../../memory/local.hpp"

#include <rund/compute/pipeline.hpp>

#include <span>

namespace rund::compute::detail {

struct PipelineState;

struct PipelineMemoryView final {
  MemoryStats summary{};
  MemoryStats shared{};
  BufferMemory scratch{};
  std::uint64_t metadata{};
  std::uint64_t referenced_resource_bytes{};
  node::accel::detail::PreparedPipelineMemory prepared{};
};

// Caller holds PipelineState::gate. The memory owner supplies one coherent
// projection; public memory and Profile owners choose their output surface.
[[nodiscard]] PipelineMemoryView pipeline_memory_view_locked(
    const PipelineState &state, std::span<PipelineStepProfile> profiles = {},
    bool include_residency_pool_buffers = true) noexcept;

} // namespace rund::compute::detail
