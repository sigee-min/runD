#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/reset/model.hpp"
#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/submission.hpp"
#include "../command/run.hpp"
#include "../kernel.hpp"
#include "../pipeline/template.hpp"
#include "../runtime/map/resources.hpp"
#include "../state.hpp"
#include "ops/model.hpp"
#include "manifest.hpp"
#include "view.hpp"
#include <memory>
#include <type_traits>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "local/resources.hpp"

struct MetalKernelContext {
  MetalAdapter *adapter = nullptr;
};
[[nodiscard]] rund::AccelCheck
ValidateMetalKernelContext(const rund::AccelDevice &pick,
                           MetalKernelContext &out);
[[nodiscard]] rund::AccelCheck
PrepareMetalStep(const rund::AccelDevice &pick, const BoundStep &step,
                 const MetalKernelOps &ops,
                 const MetalKernelImmutablePipelines *pipelines,
                 std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck
PrepareMetalSteps(const rund::AccelDevice &pick, const BoundStep *steps,
                  std::size_t step_count, KernelPreparationMode mode,
                  const KernelViewLayout *views, const RunBinds *view_binds,
                  const BackendRun *template_probe,
                  PreparedKernelTemplateRegistry *templates,
                  std::uint32_t *failed_node, MetalKernelResources &resources);
[[nodiscard]] rund::AccelCheck
EncodeMetalStep(MetalAdapter &adapter, const MetalKernelOps &ops,
                const std::shared_ptr<void> &resources,
                id<MTLComputeCommandEncoder> encoder);
[[nodiscard]] rund::AccelCheck EncodeMetalSteps(MetalAdapter &adapter,
                                                MetalKernelResources &resources,
                                                CommandRun &command);
[[nodiscard]] rund::AccelCheck
EncodeMetalResets(MetalKernelResources &resources, std::size_t step,
                  id<MTLComputeCommandEncoder> encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalStep(MetalAdapter &adapter, const MetalKernelOps &ops,
                const std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck
FinishMetalSteps(MetalAdapter &adapter, MetalKernelResources &resources,
                 rund::RuntimeStats *stats = nullptr);
#endif

} // namespace rund::node::accel::detail
