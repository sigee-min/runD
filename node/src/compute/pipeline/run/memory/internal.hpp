#pragma once

#include "../memory.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {
class Pool;
}

namespace rund::compute::detail {

// These are per-observation projections. They borrow PipelineState storage;
// neither record is retained by the pipeline or a backend.
struct PipelineSharedMemory final {
  MemoryStats stats{};
  BufferMemory scratch{};
  std::uint64_t metadata{};
  node::accel::detail::PreparedPipelineMemory prepared{};
};

struct PipelineJobMemory final {
  MemoryStats summary{};
  std::uint64_t metadata{};
};

void seal_pipeline_jobs(PipelineState &);
void seal_pipeline_shared(PipelineState &) noexcept;

[[nodiscard]] MemoryCounter
pipeline_prepared_memory(node::accel::detail::PreparedMemory memory) noexcept;
[[nodiscard]] MemoryCounter
pipeline_remaining_memory(MemoryCounter total, MemoryCounter part) noexcept;
[[nodiscard]] bool
pipeline_pool_owns_buffer(const residency::Pool *,
                          const std::shared_ptr<BufferState> &) noexcept;

[[nodiscard]] PipelineSharedMemory
measure_pipeline_shared(const PipelineState &,
                        bool include_residency_pool_buffers) noexcept;
[[nodiscard]] PipelineJobMemory
measure_pipeline_jobs(const PipelineState &, const PipelineSharedMemory &,
                      std::span<PipelineStepProfile> profiles,
                      bool include_residency_pool_buffers) noexcept;

} // namespace rund::compute::detail
