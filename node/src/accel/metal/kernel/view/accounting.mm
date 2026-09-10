#include "../view.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

PreparedMemory MetalViewMemory(const std::shared_ptr<MetalViewLowering> &view,
                               const std::uint64_t budget,
                               std::uint64_t &traffic) noexcept {
  std::uint64_t bytes = 0u;
  traffic = 0u;
  if (view != nullptr) {
    for (const MetalViewTransfer &transfer : view->transfers) {
      const std::uint64_t dense = transfer.count * transfer.element_bytes;
      traffic = ::rund::detail::counter::SaturatingAdd(traffic, dense);
      if (!transfer.planned) {
        bytes = bytes > std::numeric_limits<std::uint64_t>::max() - dense
                    ? std::numeric_limits<std::uint64_t>::max()
                    : bytes + dense;
      }
    }
  }
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = budget};
}

std::uint64_t MetalViewDispatchCount(
    const std::shared_ptr<MetalViewLowering> &view) noexcept {
  return view == nullptr ? 0u
                         : static_cast<std::uint64_t>(view->transfers.size());
}

#endif

} // namespace rund::node::accel::detail
