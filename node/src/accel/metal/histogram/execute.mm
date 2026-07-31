#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../command/run.hpp"
#include "local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteMetalHistogram(const rund::AccelDevice &pick,
                                       const rund::kernel::HistogramDesc &desc,
                                       const rund::kernel::HistogramPlan &plan,
                                       const HistogramBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepared =
      PrepareMetalHistogram(pick, desc, plan, bindings, resources);
  if (!prepared.ok) {
    return prepared;
  }
  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encoded = EncodeMetalHistogram(
      *adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submitted = FinishCommand(*adapter, command, encoded);
  if (!submitted.ok) {
    return submitted;
  }
  return FinishMetalHistogram(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
