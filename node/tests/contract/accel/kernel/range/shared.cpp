#include "local.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/vulkan/range/local.hpp"

#include <string>

namespace node_accel_contract::range {

[[nodiscard]] bool SharedHaloSourceContract() {
  using namespace rund::node::accel::detail;
  for (const rund::kernel::u32 width : kRangeWidths) {
    const RangePlan metal_range = range::PlanSourceVariant(
        RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U64,
        width, width, RangePath::SharedHalo);
    const auto requested = RangeCandidate::shared_halo(width, width);
    if (!metal_range.ok() || !requested.has_value() ||
        metal_range.candidate() != *requested) {
      return false;
    }
    const std::string metal = MetalRangeSource(range::RequireExec(metal_range));
    const std::string metal_tile =
        "threadgroup uint tile[" + std::to_string(3u * width) + "];";
    const std::size_t metal_center =
        metal.find("const uint center_value = input[");
    const std::size_t metal_left_fan =
        metal.find("left_inputs == 0u && tid == 0u", metal_center);
    const std::size_t metal_right_fan =
        metal.find("right_inputs == 0u && ulong(tid) + 1ul == active_lanes",
                   metal_left_fan);
    const std::size_t metal_barrier = metal.find(
        "threadgroup_barrier(mem_flags::mem_threadgroup);", metal_right_fan);
    const std::size_t metal_guard = metal.find(
        "if (ulong(tid) >= active_lanes) { return; }", metal_barrier);
    if (metal.find(metal_tile) == std::string::npos ||
        metal.find("* " + std::to_string(width) + "ul;") == std::string::npos ||
        metal_center == std::string::npos ||
        metal_left_fan == std::string::npos ||
        metal_right_fan == std::string::npos ||
        metal_barrier == std::string::npos ||
        metal_guard == std::string::npos ||
        metal.find("for (ulong step = 1ul;") != std::string::npos) {
      return false;
    }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
    const RangePlan vulkan_range = range::PlanSourceVariant(
        RangeSource::Vulkan, RangeOp::Sum, rund::kernel::ComputeDomain::U64,
        width, width, RangePath::SharedHalo);
    if (!vulkan_range.ok() || vulkan_range.candidate() != *requested) {
      return false;
    }
    const std::string vulkan =
        VulkanRangeSource(range::RequireExec(vulkan_range));
    const std::string vulkan_group =
        "layout(local_size_x = " + std::to_string(width) + ") in;";
    const std::string vulkan_tile =
        "shared uint64_t range_tile[" + std::to_string(3u * width) + "];";
    const std::size_t vulkan_center =
        vulkan.find("const uint64_t center_value = input_values[");
    const std::size_t vulkan_left_fan =
        vulkan.find("left_inputs == 0u && lane == 0u", vulkan_center);
    const std::size_t vulkan_right_fan = vulkan.find(
        "right_inputs == 0u && uint64_t(lane) + uint64_t(1) == active_lanes",
        vulkan_left_fan);
    const std::size_t vulkan_barrier =
        vulkan.find("barrier();", vulkan_right_fan);
    const std::size_t vulkan_guard = vulkan.find(
        "if (uint64_t(lane) >= active_lanes) { return; }", vulkan_barrier);
    const RangePlan vulkan_u32_range = range::PlanSourceVariant(
        RangeSource::Vulkan, RangeOp::Sum, rund::kernel::ComputeDomain::U32,
        width, width, RangePath::SharedHalo);
    const std::string vulkan_u32 =
        VulkanRangeSource(range::RequireExec(vulkan_u32_range));
    const std::string vulkan_extended = vulkan + '\n';
    std::uint64_t vulkan_upper = 0u;
    if (!VulkanRangeSourceBytes(range::RequireExec(vulkan_range),
                                vulkan_upper)) {
      return false;
    }
    if (vulkan.find(vulkan_group) == std::string::npos ||
        vulkan.find(vulkan_tile) == std::string::npos ||
        vulkan_center == std::string::npos ||
        vulkan_left_fan == std::string::npos ||
        vulkan_right_fan == std::string::npos ||
        vulkan_barrier == std::string::npos ||
        vulkan_guard == std::string::npos ||
        vulkan.find("for (uint64_t step = uint64_t(1);") != std::string::npos ||
        vulkan_u32.find("uint value = range_tile[first];") ==
            std::string::npos ||
        vulkan_u32.find("uint64_t value") != std::string::npos ||
        vulkan_u32.find("uint64_t(range_tile[center - step])") !=
            std::string::npos ||
        !VulkanRangeSourceMatches(range::RequireExec(vulkan_range), vulkan,
                                  SourceHash(vulkan)) ||
        VulkanRangeSourceMatches(range::RequireExec(vulkan_range),
                                 vulkan_extended,
                                 SourceHash(vulkan_extended)) ||
        VulkanRangeSourceMatches(range::RequireExec(vulkan_range), vulkan,
                                 SourceHash(vulkan) ^ 1u) ||
        vulkan.size() != vulkan_upper || vulkan_u32.empty()) {
      return false;
    }
#endif
  }

  const RangePlan metal_direct_range = range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      0u, RangePath::Direct);
  const auto direct_requested = RangeCandidate::direct_gpu(64u);
  if (!metal_direct_range.ok() || !direct_requested.has_value() ||
      metal_direct_range.candidate() != *direct_requested) {
    return false;
  }
  const std::string metal_direct =
      MetalRangeSource(range::RequireExec(metal_direct_range));
  if (metal_direct.find("threadgroup uint tile[") != std::string::npos ||
      metal_direct.find("threadgroup_barrier") != std::string::npos ||
      metal_direct.find(
          "for (ulong slot = 0ul; slot < params.window_size; ++slot)") ==
          std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const RangePlan vulkan_direct_range = range::PlanSourceVariant(
      RangeSource::Vulkan, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      0u, RangePath::Direct);
  if (!vulkan_direct_range.ok()) {
    return false;
  }
  const std::string vulkan_direct =
      VulkanRangeSource(range::RequireExec(vulkan_direct_range));
  std::uint64_t direct_upper = 0u;
  if (!VulkanRangeSourceBytes(range::RequireExec(vulkan_direct_range),
                              direct_upper)) {
    return false;
  }
  if (vulkan_direct.find("shared uint range_tile[") != std::string::npos ||
      vulkan_direct.find("barrier();") != std::string::npos ||
      vulkan_direct.find(
          "for (uint64_t slot = uint64_t(0); slot < params.window_size;") ==
          std::string::npos ||
      vulkan_direct.size() != direct_upper) {
    return false;
  }
#endif
  return true;
}

} // namespace node_accel_contract::range
