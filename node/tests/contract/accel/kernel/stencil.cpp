#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "src/accel/metal/stencil/local.hpp"
#include "src/accel/vulkan/stencil/local.hpp"
#include "stencil/local.hpp"
#include "stencil/match/run.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <iostream>

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

[[nodiscard]] bool BackendRunsStencilRemainder(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesU64(pick), "sum.u64") &&
         StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") &&
         StencilMatch(stencil::MatchesMinU32(pick), "min.u32") &&
         StencilMatch(stencil::MatchesMinI32(pick), "min.i32") &&
         StencilMatch(stencil::MatchesMaxU64(pick), "max.u64");
}

} // namespace

bool BackendRunsStencil(const rund::AccelDevice &pick) {
  return SignedStencilSourcesCarryDomainOrder() &&
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
  return BackendRunsStencilRemainder(pick);
}

} // namespace node_accel_contract
