#include "local.hpp"

#include "../match/resources.hpp"
#include "src/accel/context/internal/support.hpp"
#include "src/accel/kernel/bindings/range.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/range_aggregate/execution/projection.hpp"
#include "src/accel/stencil/range.hpp"
#include "src/accel/stencil/vulkan.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"

#include <accel/api.hpp>
#include <accel/runtime.hpp>
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>

namespace node_accel_contract::stencil::backend {
namespace {

[[nodiscard, maybe_unused]] rund::node::accel::detail::RangePlan
VulkanStencilRangePlan(const rund::AccelDevice &pick,
                       const rund::kernel::StencilPlan &plan,
                       const rund::kernel::ComputeDomain domain,
                       const bool shared_probe = false) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = ProjectStencilRange(plan, domain);
  const RangeCaps base = VulkanRangeCaps(pick);
  // Probe the supported halo limit independently of the preferred Sum
  // algorithm.
  const auto restricted = RangeCaps::gpu(
      base.source_variant(), base.legal_width_mask(),
      base.maximum_threads_per_workgroup(),
      base.shared_memory_occupancy_budget(), base.shared_memory_limit(),
      base.maximum_group_count(), base.maximum_storage_element_count(),
      base.maximum_storage_binding_bytes(),
      RangeSupportBit(RangeSupport::Direct) |
          RangeSupportBit(RangeSupport::SharedHalo));
  const RangeCaps caps =
      shared_probe ? restricted.value_or(RangeCaps::unavailable()) : base;
  return shape.has_value() ? PlanRange(*shape, caps)
                           : RangePlan::rejected("accel_kernel_graph_invalid");
}

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] RuntimeSharedProbe VulkanMaximumSharedShapeContract(
    const rund::AccelDevice &pick,
    const rund::node::accel::detail::VulkanAdapter &adapter) {
  using namespace rund::node::accel::detail;
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  if (token == nullptr || token->raw.backend.context != &adapter) {
    std::cerr << "vulkan maximum shared raw pick unavailable\n";
    return {};
  }
  std::optional<RangeCandidate> maximum{};
  for (rund::kernel::u64 radius = 256u; radius != 0u; --radius) {
    const rund::kernel::StencilPlan plan =
        rund::kernel::PlanStencil(rund::kernel::StencilDesc{
            .op = rund::kernel::StencilOp::Sum,
            .element = rund::kernel::StencilElement::U32,
            .boundary = rund::kernel::StencilBoundary::Clamp,
            .element_count = 515u,
            .radius = radius});
    const RangePlan range = VulkanStencilRangePlan(
        token->raw, plan, rund::kernel::ComputeDomain::U32, true);
    const std::optional<RangeCandidate> candidate = RangeExecCandidate(range);
    if (!candidate.has_value()) {
      std::cerr << "vulkan maximum shared probe invalid: radius=" << radius
                << '\n';
      return {};
    }
    if (!candidate->uses_shared_halo()) {
      continue;
    }
    if (candidate->radius_capacity() != radius ||
        515u % candidate->width() != 3u || range.stage_count() != 1u ||
        range.stage(0u).groups <= 1u) {
      std::cerr << "vulkan maximum shared shape mismatch: radius=" << radius
                << " width=" << candidate->width()
                << " cap=" << candidate->radius_capacity() << '\n';
      return {};
    }
    maximum = *candidate;
    break;
  }
  if (!maximum.has_value()) {
    return {RuntimeSharedProbeStatus::NoSharedCapabilityVerified, {}};
  }

  std::array<rund::kernel::u32, 515u> input{};
  auto fixture = ::node_accel_contract::stencil::match::BuildResources(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, maximum->radius_capacity(), input);
  if (!fixture.context.check.ok || !fixture.input.check.ok ||
      !fixture.output.check.ok) {
    return {};
  }
  const ContextAdmission admission = AdmitContextForSupport(fixture.context);
  std::shared_ptr<void> input_handle;
  std::shared_ptr<void> output_handle;
  if (!admission.check.ok || admission.pick == nullptr ||
      !ValidateAccelBufferForSupport(admission, fixture.input, input_handle)
           .ok ||
      !ValidateAccelBufferForSupport(admission, fixture.output, output_handle)
           .ok) {
    return {};
  }
  const rund::kernel::StencilDesc desc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = input.size(),
      .radius = maximum->radius_capacity(),
  };
  const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  const RangeBinds bindings{
      .input = &fixture.input.resident,
      .input_handle = &input_handle,
      .output = &fixture.output.resident,
      .output_handle = &output_handle,
  };
  {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    std::shared_ptr<void> first;
    const RangePlan range = VulkanStencilRangePlan(
        admission.pick->raw, plan, rund::kernel::ComputeDomain::U32, true);
    const rund::AccelCheck first_check = PrepareVulkanStencil(
        admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
        bindings, range, first);
    const auto *const first_native =
        static_cast<const VulkanRangeResources *>(first.get());
    const std::optional<RangeCandidate> first_candidate =
        first_native == nullptr ? std::nullopt
                                : RangeExecCandidate(first_native->range);
    if (!plan.ok || !range.ok() || !first_check.ok || first_native == nullptr ||
        first_native->stage_count != range.stage_count() ||
        first_native->stage_count != 1u ||
        first_native->pipelines[0u] == nullptr ||
        !first_candidate.has_value() || !first_candidate->uses_shared_halo() ||
        *first_candidate != *maximum) {
      return {};
    }
    VulkanKernelImmutablePipelines immutable{};
    immutable.kind = rund::kernel::NodeKind::Stencil;
    if (!immutable.append(first_native->pipelines[0u],
                          RangeDescriptorCount(range), 1u)) {
      return {};
    }
    rund::node::accel::ResetRuntimeStats(fixture.context.pick);
    std::shared_ptr<void> second;
    const rund::AccelCheck second_check = PrepareVulkanStencil(
        admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
        bindings, range, second, &immutable);
    const auto *const second_native =
        static_cast<const VulkanRangeResources *>(second.get());
    const std::optional<RangeCandidate> second_candidate =
        second_native == nullptr ? std::nullopt
                                 : RangeExecCandidate(second_native->range);
    const rund::RuntimeStats stats =
        rund::node::accel::ReadRuntimeStats(fixture.context.pick);
    if (!second_check.ok || second_native == nullptr ||
        !second_candidate.has_value() ||
        !second_candidate->uses_shared_halo() ||
        *second_candidate != *maximum || second_native->stage_count != 1u ||
        second_native->pipelines[0u] != first_native->pipelines[0u] ||
        !stats.outcome.ok ||
        stats.run.allocations.pipeline_compile_count != 0u ||
        stats.run.allocations.pipeline_cache_hit_count != 0u) {
      return {};
    }
  }
  if (!::node_accel_contract::stencil::MatchesCapabilitySharedBoundaryU32(
          pick, maximum->radius_capacity())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}

class VulkanStencilVariantCacheContract final {
public:
  explicit VulkanStencilVariantCacheContract(
      const rund::node::accel::detail::VulkanAdapter &) noexcept {}

  [[nodiscard]] bool Observe(const rund::AccelDevice &pick,
                             const rund::kernel::u64 element_count,
                             const rund::kernel::u64 radius,
                             const rund::kernel::ComputeDomain domain,
                             const char *const name) noexcept {
    using namespace rund::node::accel::detail;
    const rund::kernel::StencilPlan plan =
        rund::kernel::PlanStencil(rund::kernel::StencilDesc{
            .op = rund::kernel::StencilOp::Sum,
            .element = rund::kernel::StencilElement::U32,
            .boundary = rund::kernel::StencilBoundary::Clamp,
            .element_count = element_count,
            .radius = radius});
    const std::shared_ptr<PickToken> token = AdmitPick(pick);
    const RangePlan range =
        token != nullptr ? VulkanStencilRangePlan(token->raw, plan, domain)
                         : RangePlan::rejected("compute_adapter_unavailable");
    const std::optional<RangeExec> execution = RangeExec::from(range);
    if (!execution.has_value()) {
      std::cerr << "vulkan stencil variant unavailable: " << name
                << " range=" << range.reason() << '\n';
      return false;
    }
    const RangeIdentity identity = range.source_identity();
    const bool expected_hit =
        std::find(seen_.begin(), seen_.begin() + seen_count_, identity) !=
        seen_.begin() + seen_count_;
    const std::uint64_t stages = range.stage_count();
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    const bool matched = stats.outcome.ok && stages != 0u &&
                         stats.run.allocations.pipeline_compile_count ==
                             (expected_hit ? 0u : 1u) &&
                         stats.run.allocations.pipeline_cache_hit_count ==
                             (expected_hit ? 1u : 0u);
    if (!matched) {
      std::cerr << "vulkan stencil capability-derived identity mismatch: "
                << name << " width=" << execution->width()
                << " radius_cap=" << execution->shared_radius_capacity()
                << " stages=" << stages
                << " expected=" << (expected_hit ? "hit" : "cold")
                << " compile=" << stats.run.allocations.pipeline_compile_count
                << " hit=" << stats.run.allocations.pipeline_cache_hit_count
                << '\n';
      return false;
    }
    if (!expected_hit) {
      if (seen_count_ == seen_.size()) {
        return false;
      }
      seen_[seen_count_++] = identity;
    }
    return true;
  }

private:
  std::array<rund::node::accel::detail::RangeIdentity, 8u> seen_{};
  std::size_t seen_count_{};
};
#endif

} // namespace

bool RunVulkanRequired() {
  const rund::AccelDevice pick = rund::node::accel::PickAccel(
      ::node_accel_contract::primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return ::node_accel_contract::primitive::PickUnavailableReasonIsPrecise(
        pick, rund::AccelApi::Vulkan);
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  if (!StencilMatch(pick.api == rund::AccelApi::Vulkan, "pick.api")) {
    return false;
  }
  const auto *const adapter =
      static_cast<const rund::node::accel::detail::VulkanAdapter *>(
          pick.backend.context);
  if (adapter == nullptr || adapter->device == VK_NULL_HANDLE ||
      adapter->physical_device == VK_NULL_HANDLE) {
    std::cerr << "vulkan stencil adapter unavailable after successful pick\n";
    return false;
  }
  VulkanStencilVariantCacheContract variants{*adapter};
  if (!StencilMatch(::node_accel_contract::stencil::MatchesU32(pick),
                    "sum.u32") ||
      !variants.Observe(pick, 6u, 1u, rund::kernel::ComputeDomain::U32,
                        "count6.radius1")) {
    return false;
  }
  if (!StencilMatch(::node_accel_contract::stencil::MatchesSumI32(pick),
                    "sum.i32") ||
      !variants.Observe(pick, 6u, 1u, rund::kernel::ComputeDomain::I32,
                        "count6.radius1.signed-sum") ||
      !StencilMatch(::node_accel_contract::stencil::MatchesWideWindowU32(pick),
                    "sum.u32.radius2") ||
      !variants.Observe(pick, 6u, 2u, rund::kernel::ComputeDomain::U32,
                        "count6.radius2") ||
      !StencilMatch(::node_accel_contract::stencil::MatchesCount65U32(pick),
                    "sum.u32.count65") ||
      !variants.Observe(pick, 65u, 1u, rund::kernel::ComputeDomain::U32,
                        "count65.radius1") ||
      !StencilMatch(
          ::node_accel_contract::stencil::MatchesRadius64BoundaryU32(pick),
          "sum.u32.count259.radius64") ||
      !variants.Observe(pick, 259u, 64u, rund::kernel::ComputeDomain::U32,
                        "count259.radius64") ||
      !StencilMatch(
          ::node_accel_contract::stencil::MatchesPrefixDifferenceU32(pick),
          "sum.u32.prefix-difference") ||
      !variants.Observe(pick, 257u, 257u, rund::kernel::ComputeDomain::U32,
                        "count257.prefix")) {
    return false;
  }
  return RunValue(pick) && RunRange(pick) &&
         StencilMatch(
             RuntimeSharedProbeMatchesContract(
                 VulkanMaximumSharedShapeContract(pick, *adapter), "vulkan"),
             "vulkan.maximum-shared-capability");
#else
  return false;
#endif
}

} // namespace node_accel_contract::stencil::backend
