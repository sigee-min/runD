#pragma once

#include "state.hpp"

#include "../../numeric/policy.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <span>
#include <utility>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] id<MTLBuffer>
ToMetalBuffer(const MetalResidentBufferResult &buffer);
void ClearStatus(const MetalResidentBufferResult &status,
                 rund::kernel::u64 count);
[[nodiscard]] rund::AccelCheck
StatusCheck(MetalAdapter &adapter, const MetalResidentBufferResult &status,
            rund::kernel::u64 count, rund::kernel::u64 dispatches);
[[nodiscard]] rund::AccelCheck RejectElement(MetalAdapter &adapter,
                                             const char *reason);
[[nodiscard]] bool PrepareTwiddle(MetalNumericPrepared &state,
                                  const rund::kernel::TransformPlan &plan);
[[nodiscard]] rund::AccelCheck PreparedPipeline(MetalNumericPrepared &state,
                                                const char *name,
                                                NumericPolicy policy);
[[nodiscard]] rund::AccelCheck
PreparedBuffers(const rund::AccelDevice &pick, MetalNumericPrepared &state,
                std::span<MetalResidentReq> requests);
[[nodiscard]] inline rund::AccelCheck
PublishPrepared(std::shared_ptr<MetalNumericPrepared> state,
                const rund::kernel::u64 groups, const rund::kernel::u64 lanes,
                const rund::kernel::u64 dispatches,
                std::shared_ptr<void> &out) {
  state->groups = groups;
  state->threadgroup = lanes;
  state->grouped = true;
  state->dispatches = dispatches;
  out = std::move(state);
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
