#pragma once

#include "../../stencil.hpp"
#include "../../stencil/model.hpp"
#include "../../stencil/shape.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
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
    const StencilGpuShape shape, const rund::kernel::StencilElement element,
    const MetalStencilCompiledPipelineLimits limits) noexcept {
  const rund::kernel::u32 element_bytes = StencilElementBytes(element);
  if (!shape.valid() || element_bytes == 0u) {
    return MetalStencilCompiledPipelineSupport::Invalid;
  }
  if (limits.maximum_workgroup_width < shape.width()) {
    return MetalStencilCompiledPipelineSupport::Unsupported;
  }
  const rund::kernel::u64 required_bytes = shape.shared_bytes(element_bytes);
  if (!shape.uses_shared_memory()) {
    return limits.static_shared_bytes == 0u
               ? MetalStencilCompiledPipelineSupport::Supported
               : MetalStencilCompiledPipelineSupport::Invalid;
  }
  if (limits.static_shared_bytes < required_bytes) {
    return MetalStencilCompiledPipelineSupport::Invalid;
  }
  return limits.static_shared_bytes <= limits.shared_memory_limit /
                                           kStencilSharedMemoryOccupancyBudget
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
  StencilGpuShape shape{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  std::shared_ptr<void> pipeline{};
};

void DestroyMetalStencilEncodeResources(void *raw);
[[nodiscard]] std::string MetalStencilSource(rund::kernel::StencilOp op,
                                             StencilGpuShape shape);
[[nodiscard]] bool MetalStencilSourceUpperBytes(rund::kernel::StencilOp op,
                                                std::uint64_t &upper) noexcept;
[[nodiscard]] MetalStencilPipelineAttempt
CompileMetalStencilPipeline(MetalAdapter &adapter, rund::kernel::StencilOp op,
                            rund::kernel::StencilElement element,
                            rund::kernel::ComputeDomain domain,
                            StencilGpuShape shape, std::shared_ptr<void> &out);

} // namespace rund::node::accel::detail
