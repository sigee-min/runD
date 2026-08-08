#pragma once

#include "../../stencil.hpp"
#include "../../stencil/model.hpp"
#include "../../stencil/shape.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include <array>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

struct MetalStencilCompiledPipelineLimits final {
  rund::kernel::u32 maximum_workgroup_width{};
  rund::kernel::u64 static_shared_bytes{};
  rund::kernel::u64 shared_memory_limit{};
};

enum class MetalStencilCompiledPipelineSupport : std::uint8_t {
  Supported,
  Unsupported,
  Invalid,
};

[[nodiscard]] constexpr MetalStencilCompiledPipelineSupport
MetalStencilShapeCompiledPipelineSupport(
    const StencilGpuShape shape, const RangeAggregatePlan &range,
    const rund::kernel::StencilElement element,
    const MetalStencilCompiledPipelineLimits limits) noexcept {
  if (!shape.valid() || !range.ok() || StencilElementBytes(element) == 0u) {
    return MetalStencilCompiledPipelineSupport::Invalid;
  }
  if (limits.maximum_workgroup_width < shape.width()) {
    return MetalStencilCompiledPipelineSupport::Unsupported;
  }
  const rund::kernel::u64 required_bytes = StencilRangeStaticSharedBytes(range);
  if (required_bytes == 0u) {
    return limits.static_shared_bytes == 0u
               ? MetalStencilCompiledPipelineSupport::Supported
               : MetalStencilCompiledPipelineSupport::Invalid;
  }
  if (limits.static_shared_bytes < required_bytes) {
    return MetalStencilCompiledPipelineSupport::Invalid;
  }
  return limits.static_shared_bytes <=
                 limits.shared_memory_limit / kRangeAggregateSharedMemoryReserve
             ? MetalStencilCompiledPipelineSupport::Supported
             : MetalStencilCompiledPipelineSupport::Unsupported;
}

enum class MetalStencilPipelineAttemptStatus : std::uint8_t {
  Ready,
  Unsupported,
  Failed,
};

struct MetalStencilPipelineAttempt final {
  MetalStencilPipelineAttemptStatus status{
      MetalStencilPipelineAttemptStatus::Failed};
  const char *reason{"accel_metal_pipeline_unavailable"};
  std::uint64_t create_ns{};
};

struct MetalStencilEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::StencilPlan plan{};
  RangeAggregatePlan range{
      RangeAggregatePlan::rejected("compute_range_aggregate_unavailable")};
  StencilGpuShape shape{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  std::array<std::shared_ptr<void>, kRangeAggregateStageCapacity> pipelines{};
  std::array<MetalRuntimeBuffer, kRangeTemporaryCapacity> temporaries{};
  std::uint32_t stage_count{};
};

void DestroyMetalStencilEncodeResources(void *raw);
[[nodiscard]] std::string MetalStencilSource(rund::kernel::StencilOp op,
                                             StencilGpuShape shape,
                                             const RangeAggregatePlan &range);
[[nodiscard]] bool MetalStencilSourceUpperBytes(rund::kernel::StencilOp op,
                                                const RangeAggregatePlan &range,
                                                std::uint64_t &upper) noexcept;
[[nodiscard]] MetalStencilPipelineAttempt CompileMetalStencilPipeline(
    MetalAdapter &adapter, rund::kernel::StencilOp op,
    rund::kernel::StencilElement element, rund::kernel::ComputeDomain domain,
    StencilGpuShape shape, const RangeAggregatePlan &range,
    std::shared_ptr<void> &out);
[[nodiscard]] bool
MetalStencilStageScratchBindings(const MetalStencilEncodeResources &resources,
                                 std::uint32_t stage_index,
                                 const MetalRuntimeBuffer *&scratch0,
                                 const MetalRuntimeBuffer *&scratch1) noexcept;

} // namespace rund::node::accel::detail
