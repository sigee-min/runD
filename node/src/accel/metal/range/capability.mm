#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../kernel/backend/run.hpp"
#include "../../kernel/preparation.hpp"
#include "../../kernel/scratch.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../pipeline/template.hpp"
#include "../scratch.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/prepare.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::kernel::u32
MetalMaximumRangeWidth(id<MTLDevice> device) noexcept {
  return device == nil ? 0u
                       : static_cast<rund::kernel::u32>(std::min<std::uint64_t>(
                             device.maxThreadsPerThreadgroup.width,
                             std::numeric_limits<rund::kernel::u32>::max()));
}

[[nodiscard]] std::uint8_t
MetalRangeWidthMask(const rund::kernel::u32 maximum_width) noexcept {
  std::uint8_t widths = 0u;
  for (const rund::kernel::u32 width : kRangeWidths) {
    if (width <= maximum_width) {
      widths |= width == 64u    ? kRangeWidth64Bit
                : width == 128u ? kRangeWidth128Bit
                                : kRangeWidth256Bit;
    }
  }
  return widths;
}

} // namespace
#endif

RangeCaps MetalRangeCaps(const rund::AccelDevice &pick) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (!MetalPickOwnsAdapter(pick)) {
    return RangeCaps::unavailable();
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return RangeCaps::unavailable();
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter->device.get();
  const rund::kernel::u32 maximum_width = MetalMaximumRangeWidth(device);
  const std::optional<RangeCaps> capabilities = RangeCaps::gpu(
      RangeSource::Metal, MetalRangeWidthMask(maximum_width), maximum_width,
      kRangeSharedReserve, device.maxThreadgroupMemoryLength,
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<rund::kernel::u64>::max(), device.maxBufferLength,
      RangeSupportBit(RangeSupport::Direct) |
          RangeSupportBit(RangeSupport::SharedHalo) |
          RangeSupportBit(RangeSupport::PrefixDifference) |
          RangeSupportBit(RangeSupport::BlockPrefixSuffix) |
          RangeSupportBit(RangeSupport::TiledDifference));
  return capabilities.value_or(RangeCaps::unavailable());
#else
  (void)pick;
  return RangeCaps::unavailable();
#endif
}

} // namespace rund::node::accel::detail
