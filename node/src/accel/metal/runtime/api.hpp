#pragma once

#include "map/resources.hpp"

#include "../../plan/validation.hpp"
#include "../../resident/window/admission/runtime/windows.hpp"
#include "../../sequence/input/pack.hpp"
#include "../../sequence/output.hpp"
#include "../kernel.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include <span>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
void DestroyMetalMapEncodeResources(void *raw);
[[nodiscard]] bool UploadMetalBufferUncounted(const MetalRuntimeBuffer &buffer,
                                              const void *data,
                                              rund::kernel::u64 bytes);
[[nodiscard]] bool PrepareResidentBindings(
    MetalAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings, MetalResidentBindings &out);
[[nodiscard]] bool EncodeResidentWindow(
    MetalAdapter &adapter, id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings,
    const MetalResidentBindings &resident,
    std::span<const InputWindowPlan> input_plans,
    id<MTLBuffer> indirect_args = nil, NSUInteger indirect_offset = 0u);
[[nodiscard]] bool ExecuteWindows(
    MetalAdapter &adapter, const std::shared_ptr<void> &pipeline_handle,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const MetalRuntimeBuffer &param_buffer,
    const MetalResidentBindings &resident,
    std::span<const InputWindowPlan> input_plans);
[[nodiscard]] bool
ExecuteWindow(MetalAdapter &adapter,
              const std::shared_ptr<void> &pipeline_handle,
              const rund::kernel::ComputePlan &plan,
              const rund::kernel::ComputeDispatchWindow &window,
              const rund::kernel::BindingSet &bindings,
              const MetalRuntimeBuffer *param_buffer,
              std::span<const InputWindowPlan> input_plans);
#endif

} // namespace rund::node::accel::detail
