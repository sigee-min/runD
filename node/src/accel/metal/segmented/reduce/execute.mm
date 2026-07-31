#include "../../../segmented/reduce/metal.hpp"

#include "../../command/run.hpp"
#include "../../state.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteMetalSegmentedReduce(const rund::AccelDevice &pick,
                            const rund::kernel::SegmentedReduceDesc &desc,
                            const rund::kernel::SegmentedReducePlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            const SegmentedReduceBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return {false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  rund::AccelCheck check = PrepareMetalSegmentedReduce(pick, desc, plan, domain,
                                                       bindings, resources);
  if (!check.ok) {
    return check;
  }
  CommandRun command{};
  check = OpenCommand(*adapter, command);
  if (!check.ok) {
    return check;
  }
  check = EncodeMetalSegmentedReduce(*adapter, resources,
                                     (__bridge void *)command.encoder);
  check = FinishCommand(*adapter, command, check);
  return check.ok ? FinishMetalSegmentedReduce(*adapter, resources) : check;
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  return {false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
