#include "local.hpp"

#include "source/block.hpp"
#include "source/offset.hpp"
#include "source/prefix.hpp"
#include "source/prelude.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
std::string
VulkanSegmentedScanSource(const rund::kernel::SegmentedScanElement element,
                          const rund::kernel::ComputeDomain domain,
                          const std::string_view phase) {
  std::string source;
  AppendSegmentedPrelude(source, element, domain, phase != "prefix");
  if (phase == "prefix") {
    AppendSegmentedPrefix(source, element);
  } else if (phase == "offset") {
    AppendSegmentedOffset(source, element);
  } else {
    AppendSegmentedBlock(source, element);
  }
  source += "}\n";
  return source;
}
#endif

} // namespace rund::node::accel::detail
