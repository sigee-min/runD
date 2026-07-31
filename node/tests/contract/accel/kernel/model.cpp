#include "src/accel/metal/gather/local.hpp"
#include "src/accel/metal/histogram/local.hpp"
#include "src/accel/metal/partition/local.hpp"
#include "src/accel/metal/scatter/local.hpp"
#include "src/accel/metal/segmented/local.hpp"
#include "src/accel/metal/stencil/local.hpp"
#include "src/accel/vulkan/gather/local.hpp"
#include "src/accel/vulkan/histogram/local.hpp"
#include "src/accel/vulkan/partition/local.hpp"
#include "src/accel/vulkan/scatter/local.hpp"
#include "src/accel/vulkan/segmented/local.hpp"
#include "src/accel/vulkan/stencil/local.hpp"

#include <string>
#include <string_view>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool OneLayout(const std::string_view source,
                             const std::string_view layout) {
  const std::size_t first = source.find(layout);
  return first != std::string_view::npos &&
         source.find(layout, first + layout.size()) == std::string_view::npos;
}

[[nodiscard]] bool MetalLayoutsMatchHost() {
  using namespace rund::node::accel::detail;
  return OneLayout(MetalGatherSource(), "struct GatherParams {\n"
                                        "  ulong element_count;\n"
                                        "  ulong source_count;\n"
                                        "  uint count_source;\n"
                                        "  uint reserved;\n"
                                        "};") &&
         OneLayout(MetalHistogramSource(), "struct HistogramParams {\n"
                                           "  ulong element_count;\n"
                                           "  ulong bin_count;\n"
                                           "};") &&
         OneLayout(MetalPartitionSource(), "struct PartitionParams {\n"
                                           "  ulong element_count;\n"
                                           "};") &&
         OneLayout(MetalScatterSource(), "struct ScatterParams {\n"
                                         "  ulong element_count;\n"
                                         "  ulong output_count;\n"
                                         "};") &&
         OneLayout(MetalSegmentedScanSource(), "struct SegmentedScanParams {\n"
                                               "  ulong element_count;\n"
                                               "  ulong block_size;\n"
                                               "  ulong block_count;\n"
                                               "  uint inclusive;\n"
                                               "  uint reserved;\n"
                                               "};") &&
         OneLayout(
             MetalSegmentedScanSource(),
             "    if (bad != 0u) { \\\n"
             "      atomic_fetch_max_explicit(&segment_status, bad, "
             "memory_order_relaxed); \\\n"
             "    } \\\n"
             "    threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
             "    if (lane == 0u) { \\") &&
         OneLayout(MetalStencilSource(rund::kernel::StencilOp::Sum),
                   "struct StencilParams {\n"
                   "  ulong element_count;\n"
                   "  ulong radius;\n"
                   "};");
}

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool VulkanLayoutsMatchHost() {
  using namespace rund::node::accel::detail;
  return OneLayout(
             VulkanGatherSource(rund::kernel::GatherElement::U32, false),
             "layout(set = 0, binding = 0, std430) readonly buffer Params {\n"
             "  uint64_t element_count;\n"
             "  uint64_t source_count;\n"
             "  uint count_source;\n"
             "  uint reserved;\n"
             "} params;") &&
         OneLayout(
             VulkanHistogramSource(false),
             "layout(set = 0, binding = 0, std430) readonly buffer Params {\n"
             "  uint64_t element_count;\n"
             "  uint64_t bin_count;\n"
             "} params;") &&
         OneLayout(
             VulkanPartitionSource(
                 rund::node::accel::detail::PartitionStage::Classify,
                 sizeof(rund::kernel::u32), sizeof(rund::kernel::u32)),
             "layout(set = 0, binding = 0, std430) readonly buffer Params {\n"
             "  uint64_t element_count;\n"
             "} params;") &&
         OneLayout(
             VulkanScatterSource(rund::kernel::ScatterElement::U32),
             "layout(set = 0, binding = 0, std430) readonly buffer Params {\n"
             "  uint64_t element_count;\n"
             "  uint64_t output_count;\n"
             "} params;") &&
         OneLayout(
             VulkanSegmentedScanSource(rund::kernel::SegmentedScanElement::U32,
                                       rund::kernel::ComputeDomain::U32,
                                       "block"),
             "layout(set = 0, binding = 0, std430) readonly buffer Params {\n"
             "  uint64_t element_count; uint64_t block_size;\n"
             "  uint64_t block_count; uint inclusive; uint reserved;\n"
             "} params;") &&
         OneLayout(
             VulkanSegmentedScanSource(
                 rund::kernel::SegmentedScanElement::U32,
                 rund::kernel::ComputeDomain::U32, "block"),
             "    if (bad != 0u) { atomicMax(segment_status, bad); }\n"
             "    barrier();\n"
             "    if (lane == 0u) {\n") &&
         OneLayout(
             VulkanStencilSource(rund::kernel::StencilOp::Sum,
                                 rund::kernel::StencilElement::U32,
                                 rund::kernel::ComputeDomain::U32),
             "layout(set = 0, binding = 0, std430) readonly buffer Params {\n"
             "  uint64_t element_count;\n"
             "  uint64_t radius;\n"
             "} params;");
}
#endif

} // namespace

bool BackendParameterModelsMatchSources() {
  if (!MetalLayoutsMatchHost()) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return VulkanLayoutsMatchHost();
#else
  return true;
#endif
}

} // namespace node_accel_contract
