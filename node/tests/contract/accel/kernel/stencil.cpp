#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "src/accel/metal/stencil/local.hpp"
#include "src/accel/stencil/shape.hpp"
#include "src/accel/vulkan/stencil/local.hpp"
#include "stencil/local.hpp"
#include "stencil/match/run.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <algorithm>
#include <iostream>
#include <limits>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool StencilMatch(const bool ok, const char *const name) {
  if (ok) {
    return true;
  }
  std::cerr << "stencil backend match failed: " << name << '\n';
  return false;
}

[[nodiscard]] bool SignedStencilSourcesCarryDomainOrder() {
  const std::string metal = rund::node::accel::detail::MetalStencilSource(
      rund::kernel::StencilOp::Min);
  if (metal.find("rund_compute_stencil_min_i32") == std::string::npos ||
      metal.find("device const int* input") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string vulkan = rund::node::accel::detail::VulkanStencilSource(
      rund::kernel::StencilOp::Min, rund::kernel::StencilElement::U32,
      rund::kernel::ComputeDomain::I32);
  if (vulkan.find("int value = int(input_values[gid])") == std::string::npos) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] constexpr bool SharedStencilTrafficModelIsStrict() noexcept {
  using namespace rund::node::accel::detail;
  constexpr rund::kernel::u64 u32_groups =
      std::numeric_limits<rund::kernel::u32>::max();
  constexpr rund::kernel::u64 u32_group_elements =
      u32_groups * kStencilPhysicalGroupWidth;
  constexpr rund::kernel::u64 vulkan_u32_groups =
      StencilPhysicalGroupCount(std::numeric_limits<rund::kernel::u32>::max());
  if (kStencilPhysicalGroupWidth != 256u || kStencilSharedRadiusCap != 8u ||
      kStencilSharedElementCapacity != 272u ||
      kStencilSharedBytesMax != 2176u || StencilPhysicalGroupCount(0u) != 0u ||
      StencilPhysicalGroupCount(1u) != 1u ||
      StencilPhysicalGroupCount(256u) != 1u ||
      StencilPhysicalGroupCount(257u) != 2u ||
      !StencilPhysicalGroupsFit(7u * kStencilPhysicalGroupWidth, 7u) ||
      StencilPhysicalGroupsFit(7u * kStencilPhysicalGroupWidth + 1u, 7u) ||
      !StencilPhysicalGroupsFit(u32_group_elements, u32_groups) ||
      StencilPhysicalGroupsFit(u32_group_elements + 1u, u32_groups) ||
      StencilPhysicalGroupsFit(u32_group_elements + 1u, u32_groups + 1u) ||
      !StencilVulkanDispatchFits(std::numeric_limits<rund::kernel::u32>::max(),
                                 vulkan_u32_groups) ||
      StencilVulkanDispatchFits(
          static_cast<rund::kernel::u64>(
              std::numeric_limits<rund::kernel::u32>::max()) +
              1u,
          vulkan_u32_groups + 1u) ||
      StencilPhysicalGroupsFit(1u, 0u)) {
    return false;
  }
  constexpr rund::kernel::u64 count_limit =
      2u * kStencilPhysicalGroupWidth + kStencilSharedRadiusCap;
  for (rund::kernel::u64 count = 1u; count <= count_limit; ++count) {
    const rund::kernel::u64 groups = StencilPhysicalGroupCount(count);
    for (rund::kernel::u64 group = 0u; group < groups; ++group) {
      const rund::kernel::u64 base = group * kStencilPhysicalGroupWidth;
      const rund::kernel::u32 active =
          static_cast<rund::kernel::u32>(std::min<rund::kernel::u64>(
              count - base, kStencilPhysicalGroupWidth));
      const rund::kernel::u64 end = base + active;
      for (rund::kernel::u32 radius = 1u;
           radius <= kStencilSharedRadiusCap && radius <= count; ++radius) {
        const rund::kernel::u64 union_begin =
            base < radius ? 0u : base - radius;
        const rund::kernel::u64 union_end =
            std::min<rund::kernel::u64>(count, end + radius);
        const rund::kernel::u64 shared =
            StencilSharedGlobalReads(count, base, active, radius);
        if (shared != union_end - union_begin ||
            shared >= StencilDirectGlobalReads(active, radius)) {
          return false;
        }
      }
    }
  }
  return StencilDirectGlobalReads(256u, 1u) == 768u &&
         StencilSharedGlobalReads(256u, 0u, 256u, 1u) == 256u &&
         StencilSharedGlobalReads(257u, 0u, 256u, 1u) == 257u &&
         StencilSharedGlobalReads(520u, 256u, 256u, 1u) == 258u &&
         StencilDirectGlobalReads(256u, 8u) == 4352u &&
         StencilSharedGlobalReads(520u, 256u, 256u, 8u) == 272u &&
         StencilSharedGlobalReads(259u, 0u, 256u, 8u) == 259u &&
         !StencilUsesSharedHalo(0u) && StencilUsesSharedHalo(1u) &&
         StencilUsesSharedHalo(8u) && !StencilUsesSharedHalo(9u);
}

static_assert(SharedStencilTrafficModelIsStrict());

[[nodiscard]] bool StencilShapeRejectsOnlyOverlappingStorage() {
  using namespace rund::node::accel::detail;
  constexpr rund::kernel::StencilDesc desc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = 4u,
      .radius = 1u,
  };
  constexpr rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  static_assert(plan.ok);
  rund::kernel::ResidentBufferRef input{
      .id = 41u,
      .bytes = 32u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  rund::kernel::ResidentBufferRef output{
      .id = input.id,
      .bytes = input.bytes,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const std::shared_ptr<void> owner = std::make_shared<int>(1);
  const StencilBinds bindings{
      .input = &input,
      .input_handle = &owner,
      .output = &output,
      .output_handle = &owner,
  };
  if (StencilShapeOk(desc, plan, bindings)) {
    return false;
  }
  output.offset_bytes = 16u;
  return StencilShapeOk(desc, plan, bindings);
}

[[nodiscard]] bool StencilSourcesCarryConvergentSharedHalo() {
  using namespace rund::node::accel::detail;
  const std::string metal = MetalStencilSource(rund::kernel::StencilOp::Sum);
  const std::size_t metal_tile = metal.find("threadgroup uint tile[272];");
  const std::size_t metal_select =
      metal.find("params.radius <= 8ul)", metal_tile);
  const std::size_t metal_center =
      metal.find("const uint center_value = input[", metal_select);
  const std::size_t metal_left_fan =
      metal.find("left_inputs == 0u && tid == 0u", metal_center);
  const std::size_t metal_right_fan = metal.find(
      "right_inputs == 0u && ulong(tid) + 1ul == active_lanes", metal_left_fan);
  const std::size_t metal_barrier = metal.find(
      "threadgroup_barrier(mem_flags::mem_threadgroup);", metal_right_fan);
  const std::size_t metal_guard =
      metal.find("if (ulong(tid) >= active_lanes) { return; }", metal_barrier);
  const std::size_t metal_index =
      metal.find("const ulong i = group_base + ulong(tid);", metal_guard);
  const std::size_t metal_fallback = metal.find(
      "for (ulong step = 1ul; step <= params.radius; ++step)", metal_index);
  if (metal_tile == std::string::npos || metal_select == std::string::npos ||
      metal_center == std::string::npos ||
      metal_left_fan == std::string::npos ||
      metal_right_fan == std::string::npos ||
      metal_barrier == std::string::npos || metal_guard == std::string::npos ||
      metal_index == std::string::npos || metal_fallback == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string vulkan = VulkanStencilSource(
      rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U64,
      rund::kernel::ComputeDomain::U64);
  const std::size_t vulkan_group =
      vulkan.find("layout(local_size_x = 256) in;");
  const std::size_t vulkan_tile =
      vulkan.find("shared uint64_t stencil_tile[272];", vulkan_group);
  const std::size_t vulkan_select =
      vulkan.find("params.radius <= uint64_t(8)", vulkan_tile);
  const std::size_t vulkan_center =
      vulkan.find("const uint64_t center_value = input_values[", vulkan_select);
  const std::size_t vulkan_left_fan =
      vulkan.find("left_inputs == 0u && lane == 0u", vulkan_center);
  const std::size_t vulkan_right_fan = vulkan.find(
      "right_inputs == 0u && uint64_t(lane) + uint64_t(1) == active_lanes",
      vulkan_left_fan);
  const std::size_t vulkan_barrier =
      vulkan.find("barrier();", vulkan_right_fan);
  const std::size_t vulkan_guard = vulkan.find(
      "if (uint64_t(lane) >= active_lanes) { return; }", vulkan_barrier);
  const std::size_t vulkan_index = vulkan.find(
      "const uint gid = uint(group_base + uint64_t(lane));", vulkan_guard);
  const std::size_t vulkan_fallback = vulkan.find(
      "for (uint64_t step = uint64_t(1); step <= params.radius; ++step)",
      vulkan_index);
  if (vulkan_group == std::string::npos || vulkan_tile == std::string::npos ||
      vulkan_select == std::string::npos ||
      vulkan_center == std::string::npos ||
      vulkan_left_fan == std::string::npos ||
      vulkan_right_fan == std::string::npos ||
      vulkan_barrier == std::string::npos ||
      vulkan_guard == std::string::npos || vulkan_index == std::string::npos ||
      vulkan_fallback == std::string::npos) {
    return false;
  }
  const std::string vulkan_u32 = VulkanStencilSource(
      rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
      rund::kernel::ComputeDomain::U32);
  if (vulkan_u32.find("uint value = stencil_tile[center];") ==
          std::string::npos ||
      vulkan_u32.find("uint value = input_values[gid];") == std::string::npos ||
      vulkan_u32.find("uint64_t value") != std::string::npos ||
      vulkan_u32.find("uint64_t(stencil_tile[center - step])") !=
          std::string::npos ||
      vulkan_u32.find("uint64_t(input_values[left])") != std::string::npos) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool BackendRunsStencilShapeCases(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") &&
         StencilMatch(stencil::MatchesSharedBoundaryU32(pick),
                      "sum.u32.radius8.shared-boundary") &&
         StencilMatch(stencil::MatchesDirectFallbackU32(pick),
                      "sum.u32.radius9.direct-fallback");
}

[[nodiscard]] bool BackendRunsStencilValueCases(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesU64(pick), "sum.u64") &&
         StencilMatch(stencil::MatchesMinU32(pick), "min.u32") &&
         StencilMatch(stencil::MatchesMinI32(pick), "min.i32") &&
         StencilMatch(stencil::MatchesMaxU64(pick), "max.u64");
}

[[nodiscard]] bool BackendRunsStencilRemainder(const rund::AccelDevice &pick) {
  return BackendRunsStencilShapeCases(pick) &&
         BackendRunsStencilValueCases(pick);
}

[[nodiscard]] bool
VulkanStencilRunWasCacheHitOnly(const rund::AccelDevice &pick,
                                const char *const name) {
  const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
  if (stats.ok && stats.pipeline_compile_count == 0u &&
      stats.pipeline_cache_hit_count == 1u) {
    return true;
  }
  std::cerr << "vulkan stencil source-shape cache mismatch: " << name
            << " compile=" << stats.pipeline_compile_count
            << " hit=" << stats.pipeline_cache_hit_count << '\n';
  return false;
}

} // namespace

bool BackendRunsStencil(const rund::AccelDevice &pick) {
  return SharedStencilTrafficModelIsStrict() &&
         StencilShapeRejectsOnlyOverlappingStorage() &&
         StencilSourcesCarryConvergentSharedHalo() &&
         SignedStencilSourcesCarryDomainOrder() &&
         StencilMatch(stencil::MatchesU32(pick), "sum.u32") &&
         BackendRunsStencilRemainder(pick);
}

bool RequiredMetalRunsStencil() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  return pick.api == rund::AccelApi::Metal && BackendRunsStencil(pick);
}

bool RequiredVulkanRunsStencil() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  if (pick.api != rund::AccelApi::Vulkan ||
      !StencilSourcesCarryConvergentSharedHalo() ||
      !SignedStencilSourcesCarryDomainOrder() ||
      !StencilMatch(stencil::MatchesU32(pick), "sum.u32")) {
    return false;
  }
  const rund::RuntimeStats unsigned_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!unsigned_stats.ok || unsigned_stats.pipeline_compile_count != 1u ||
      unsigned_stats.pipeline_cache_hit_count != 0u) {
    std::cerr << "vulkan unsigned sum stencil cold cache mismatch: compile="
              << unsigned_stats.pipeline_compile_count
              << " hit=" << unsigned_stats.pipeline_cache_hit_count << '\n';
    return false;
  }
  if (!StencilMatch(stencil::MatchesSumI32(pick), "sum.i32")) {
    return false;
  }
  const rund::RuntimeStats signed_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!signed_stats.ok || signed_stats.pipeline_compile_count != 0u ||
      signed_stats.pipeline_cache_hit_count != 1u) {
    std::cerr << "vulkan signed sum stencil shared source mismatch: compile="
              << signed_stats.pipeline_compile_count
              << " hit=" << signed_stats.pipeline_cache_hit_count << '\n';
    return false;
  }
  if (!StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") ||
      !VulkanStencilRunWasCacheHitOnly(pick, "radius2") ||
      !StencilMatch(stencil::MatchesSharedBoundaryU32(pick),
                    "sum.u32.radius8.shared-boundary") ||
      !VulkanStencilRunWasCacheHitOnly(pick, "count259.radius8") ||
      !StencilMatch(stencil::MatchesDirectFallbackU32(pick),
                    "sum.u32.radius9.direct-fallback") ||
      !VulkanStencilRunWasCacheHitOnly(pick, "count17.radius9")) {
    return false;
  }
  return BackendRunsStencilValueCases(pick);
}

} // namespace node_accel_contract
