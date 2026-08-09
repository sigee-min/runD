#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "../range/local.hpp"
#include "contract.hpp"
#include "local.hpp"
#include "match/run.hpp"
#include "src/accel/context/internal/support.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/metal/pipeline/cache.hpp"
#include "src/accel/metal/pipeline/template.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/stencil/shape.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool StencilMatch(const bool ok, const char *const name) {
  if (ok) {
    return true;
  }
  std::cerr << "stencil backend match failed: " << name << '\n';
  return false;
}

[[nodiscard]] rund::node::accel::detail::RangePlan
MetalStencilRangePlan(const rund::AccelDevice &pick,
                      const rund::kernel::StencilPlan &plan,
                      const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = StencilRangeShape(plan, domain);
  return shape.has_value() ? PlanRange(*shape, MetalRangeCaps(pick))
                           : RangePlan::rejected("accel_kernel_graph_invalid");
}

[[nodiscard, maybe_unused]] rund::node::accel::detail::RangePlan
VulkanStencilRangePlan(const rund::AccelDevice &pick,
                       const rund::kernel::StencilPlan &plan,
                       const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = StencilRangeShape(plan, domain);
  return shape.has_value() ? PlanRange(*shape, VulkanRangeCaps(pick))
                           : RangePlan::rejected("accel_kernel_graph_invalid");
}

enum class RuntimeSharedProbeStatus : std::uint8_t {
  Failed,
  SharedVerified,
  NoSharedCapabilityVerified,
};

struct RuntimeSharedProbe final {
  RuntimeSharedProbeStatus status{RuntimeSharedProbeStatus::Failed};
  std::optional<rund::node::accel::detail::RangeCandidate> candidate{};
};

[[nodiscard]] constexpr std::optional<rund::node::accel::detail::RangeCandidate>
RangeExecCandidate(const rund::node::accel::detail::RangePlan &plan) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value()
             ? std::optional<RangeCandidate>{execution->plan().candidate()}
             : std::nullopt;
}

#if defined(__APPLE__)
[[nodiscard]] RuntimeSharedProbe
MetalMaximumSharedShapeContract(const rund::AccelDevice &pick) {
  using namespace rund::node::accel::detail;
  std::array<rund::kernel::u32, 515u> input{};
  auto fixture = stencil::match::BuildResources(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 1u, input);
  if (!fixture.context.check.ok || !fixture.input.check.ok ||
      !fixture.output.check.ok) {
    std::cerr << "metal stencil immutable fixture unavailable: context="
              << fixture.context.check.reason
              << " input=" << fixture.input.check.reason
              << " output=" << fixture.output.check.reason << '\n';
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
  const RangeBinds bindings{
      .input = &fixture.input.resident,
      .input_handle = &input_handle,
      .output = &fixture.output.resident,
      .output_handle = &output_handle,
  };
  std::optional<RangeCandidate> maximum{};
  {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    std::shared_ptr<void> first;
    rund::kernel::StencilDesc selected_desc{};
    rund::kernel::StencilPlan selected_plan{};
    for (rund::kernel::u64 radius = 256u; radius != 0u; --radius) {
      const rund::kernel::StencilDesc desc{
          .op = rund::kernel::StencilOp::Sum,
          .element = rund::kernel::StencilElement::U32,
          .boundary = rund::kernel::StencilBoundary::Clamp,
          .element_count = input.size(),
          .radius = radius,
      };
      const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
      const RangePlan range = MetalStencilRangePlan(
          admission.pick->raw, plan, rund::kernel::ComputeDomain::U32);
      std::shared_ptr<void> candidate;
      const rund::AccelCheck check = PrepareMetalStencil(
          admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
          bindings, range, candidate);
      const auto *const native =
          static_cast<const MetalRangeResources *>(candidate.get());
      const std::optional<RangeCandidate> native_candidate =
          native == nullptr ? std::nullopt : RangeExecCandidate(native->range);
      if (!plan.ok || !range.ok() || !check.ok || native == nullptr ||
          !native_candidate.has_value() ||
          native->stage_count != range.stage_count() ||
          native->stage_count == 0u || native->pipelines[0u] == nullptr) {
        std::cerr << "metal maximum shared probe failed: radius=" << radius
                  << " check=" << check.ok << " reason=" << check.reason
                  << " native=" << (native != nullptr) << '\n';
        return {};
      }
      if (!native_candidate->uses_shared_halo()) {
        continue;
      }
      if (native_candidate->radius_capacity() != radius ||
          input.size() % native_candidate->width() != 3u ||
          range.stage_count() != 1u || range.stage(0u).groups <= 1u) {
        std::cerr << "metal maximum shared shape mismatch: radius=" << radius
                  << " width=" << native_candidate->width()
                  << " cap=" << native_candidate->radius_capacity() << '\n';
        return {};
      }
      maximum = *native_candidate;
      selected_desc = desc;
      selected_plan = plan;
      first = std::move(candidate);
      break;
    }
    if (!maximum.has_value()) {
      return {RuntimeSharedProbeStatus::NoSharedCapabilityVerified, {}};
    }

    const auto *const first_native =
        static_cast<const MetalRangeResources *>(first.get());
    MetalKernelImmutablePipelines immutable{};
    if (first_native->stage_count != 1u ||
        first_native->pipelines[0u] == nullptr) {
      return {};
    }
    immutable.stages[0u] = first_native->pipelines[0u];
    immutable.count = 1u;
    rund::node::accel::ResetRuntimeStats(fixture.context.pick);
    std::shared_ptr<void> second;
    const RangePlan selected_range = MetalStencilRangePlan(
        admission.pick->raw, selected_plan, rund::kernel::ComputeDomain::U32);
    const rund::AccelCheck second_check =
        PrepareMetalStencil(admission.pick->raw, selected_desc, selected_plan,
                            rund::kernel::ComputeDomain::U32, bindings,
                            selected_range, second, &immutable);
    const auto *const second_native =
        static_cast<const MetalRangeResources *>(second.get());
    const std::optional<RangeCandidate> second_candidate =
        second_native == nullptr ? std::nullopt
                                 : RangeExecCandidate(second_native->range);
    const rund::RuntimeStats stats =
        rund::node::accel::ReadRuntimeStats(fixture.context.pick);
    const bool matched =
        second_check.ok && second_native != nullptr &&
        second_candidate.has_value() && second_candidate->uses_shared_halo() &&
        *second_candidate == *maximum && stats.outcome.ok &&
        stats.run.allocations.pipeline_compile_count == 0u &&
        stats.run.allocations.pipeline_cache_hit_count == 0u &&
        second_native->stage_count == 1u &&
        second_native->pipelines[0u] == first_native->pipelines[0u];
    if (!matched) {
      std::cerr << "metal maximum shared immutable borrow mismatch: check="
                << second_check.ok << " reason=" << second_check.reason
                << " native=" << (second_native != nullptr)
                << " compile=" << stats.run.allocations.pipeline_compile_count
                << " hit=" << stats.run.allocations.pipeline_cache_hit_count
                << '\n';
      return {};
    }
  }
  if (!stencil::MatchesCapabilitySharedBoundaryU32(
          pick, maximum->radius_capacity())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}
#endif

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
        token->raw, plan, rund::kernel::ComputeDomain::U32);
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
  auto fixture = stencil::match::BuildResources(
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
        admission.pick->raw, plan, rund::kernel::ComputeDomain::U32);
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
  if (!stencil::MatchesCapabilitySharedBoundaryU32(
          pick, maximum->radius_capacity())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}
#endif

[[nodiscard]] bool BackendRunsStencilShapeCases(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") &&
         StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") &&
         StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                      "sum.u32.count259.radius64") &&
         StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                      "sum.u32.prefix-difference");
}

[[nodiscard]] bool
RuntimeSharedProbeMatchesContract(const RuntimeSharedProbe probe,
                                  const char *const backend) {
  switch (probe.status) {
  case RuntimeSharedProbeStatus::SharedVerified:
    if (probe.candidate.has_value() && probe.candidate->uses_shared_halo() &&
        probe.candidate->radius_capacity() != 0u) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::NoSharedCapabilityVerified:
    if (!probe.candidate.has_value()) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::Failed:
    break;
  }
  std::cerr << backend << " runtime maximum shared contract failed: status="
            << static_cast<unsigned>(probe.status) << " width="
            << (probe.candidate.has_value() ? probe.candidate->width() : 0u)
            << " cap="
            << (probe.candidate.has_value() ? probe.candidate->radius_capacity()
                                            : 0u)
            << '\n';
  return false;
}

[[nodiscard]] bool BackendRunsStencilValueCases(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesU64(pick), "sum.u64") &&
         StencilMatch(stencil::MatchesMinU32(pick), "min.u32") &&
         StencilMatch(stencil::MatchesMinI32(pick), "min.i32") &&
         StencilMatch(stencil::MatchesMaxU64(pick), "max.u64") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMinI32(pick),
                      "min.i32.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMaxI32(pick),
                      "max.i32.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMinU64(pick),
                      "min.u64.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMaxU64(pick),
                      "max.u64.block-prefix-suffix");
}

[[nodiscard]] bool BackendRunsRangeParity(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesForcedPrefixDifferenceU32(pick),
                      "sum.u32.direct-prefix-difference") &&
         StencilMatch(stencil::MatchesForcedPrefixDifferenceU64(pick),
                      "sum.u64.direct-prefix-difference") &&
         StencilMatch(stencil::MatchesDeepPrefixHierarchyU32(pick),
                      "sum.u32.prefix-hierarchy") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMinI32(pick),
                      "min.i32.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMaxI32(pick),
                      "max.i32.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMinU64(pick),
                      "min.u64.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMaxU64(pick),
                      "max.u64.direct-block-prefix-suffix");
}

[[nodiscard]] bool BackendRunsStencilRemainder(const rund::AccelDevice &pick) {
  return BackendRunsStencilShapeCases(pick) &&
         BackendRunsStencilValueCases(pick);
}

class MetalStencilVariantCacheContract final {
public:
  explicit MetalStencilVariantCacheContract(
      rund::node::accel::detail::MetalAdapter &adapter)
      : adapter_(adapter) {
    valid_ = Snapshot();
  }

  [[nodiscard]] bool Observe(const rund::AccelDevice &pick,
                             const char *const name) {
    const std::size_t before = seen_count_;
    if (!valid_ || !Snapshot()) {
      return false;
    }
    const std::size_t inserted = seen_count_ - before;
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    const bool cold = inserted != 0u;
    const bool matched =
        stats.outcome.ok &&
        (cold ? stats.run.allocations.pipeline_compile_count >= inserted
              : stats.run.allocations.pipeline_cache_hit_count != 0u);
    if (!matched) {
      std::cerr << "metal stencil capability-derived identity mismatch: "
                << name << " new_variants=" << inserted
                << " compile=" << stats.run.allocations.pipeline_compile_count
                << " hit=" << stats.run.allocations.pipeline_cache_hit_count
                << '\n';
      return false;
    }
    observed_cold_ = observed_cold_ || cold;
    observed_hit_ = observed_hit_ || !cold;
    return true;
  }

  [[nodiscard]] bool complete() const noexcept {
    return valid_ && observed_cold_ && observed_hit_;
  }

private:
  [[nodiscard]] static bool Tracked(const std::string &name) noexcept {
    return name.find("range.aggregate.") != std::string::npos;
  }

  [[nodiscard]] bool Seen(const std::string &name) const noexcept {
    return std::find(seen_.begin(), seen_.begin() + seen_count_, name) !=
           seen_.begin() + seen_count_;
  }

  [[nodiscard]] bool Snapshot() {
    std::lock_guard<std::mutex> lock{adapter_.mutex};
    for (const rund::node::accel::detail::MetalNamedPipeline &pipeline :
         adapter_.named_pipelines) {
      if (!Tracked(pipeline.name) || Seen(pipeline.name)) {
        continue;
      }
      if (seen_count_ == seen_.size()) {
        valid_ = false;
        return false;
      }
      seen_[seen_count_++] = pipeline.name;
    }
    return true;
  }

  rund::node::accel::detail::MetalAdapter &adapter_;
  std::array<std::string, 16u> seen_{};
  std::size_t seen_count_{};
  bool valid_{};
  bool observed_cold_{};
  bool observed_hit_{};
};

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
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

bool stencil::RunBackend(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::StorageContract(), "shape.storage") &&
         StencilMatch(stencil::MatchesU32(pick), "sum.u32") &&
         BackendRunsStencilRemainder(pick);
}

bool stencil::RunRequiredMetal() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  if (!StencilMatch(pick.api == rund::AccelApi::Metal, "pick.api") ||
      !StencilMatch(stencil::StorageContract(), "shape.storage")) {
    return false;
  }
  auto *const adapter = static_cast<rund::node::accel::detail::MetalAdapter *>(
      pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    std::cerr << "metal stencil adapter unavailable after successful pick\n";
    return false;
  }
  MetalStencilVariantCacheContract variants{*adapter};
  if (!StencilMatch(stencil::MatchesU32(pick), "sum.u32") ||
      !variants.Observe(pick, "count6.radius1") ||
      !StencilMatch(stencil::MatchesSumI32(pick), "sum.i32") ||
      !variants.Observe(pick, "count6.radius1.signed-sum") ||
      !StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") ||
      !variants.Observe(pick, "count6.radius2") ||
      !StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") ||
      !variants.Observe(pick, "count65.radius1") ||
      !StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                    "sum.u32.count259.radius64") ||
      !variants.Observe(pick, "count259.radius64") ||
      !StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                    "sum.u32.prefix-difference") ||
      !variants.Observe(pick, "count257.prefix") || !variants.complete()) {
    return false;
  }
#if defined(__APPLE__)
  return BackendRunsStencilValueCases(pick) && BackendRunsRangeParity(pick) &&
         StencilMatch(RuntimeSharedProbeMatchesContract(
                          MetalMaximumSharedShapeContract(pick), "metal"),
                      "metal.maximum-shared-capability");
#else
  return false;
#endif
}

bool stencil::RunRequiredVulkan() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
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
  if (!StencilMatch(stencil::MatchesU32(pick), "sum.u32") ||
      !variants.Observe(pick, 6u, 1u, rund::kernel::ComputeDomain::U32,
                        "count6.radius1")) {
    return false;
  }
  if (!StencilMatch(stencil::MatchesSumI32(pick), "sum.i32") ||
      !variants.Observe(pick, 6u, 1u, rund::kernel::ComputeDomain::I32,
                        "count6.radius1.signed-sum") ||
      !StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") ||
      !variants.Observe(pick, 6u, 2u, rund::kernel::ComputeDomain::U32,
                        "count6.radius2") ||
      !StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") ||
      !variants.Observe(pick, 65u, 1u, rund::kernel::ComputeDomain::U32,
                        "count65.radius1") ||
      !StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                    "sum.u32.count259.radius64") ||
      !variants.Observe(pick, 259u, 64u, rund::kernel::ComputeDomain::U32,
                        "count259.radius64") ||
      !StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                    "sum.u32.prefix-difference") ||
      !variants.Observe(pick, 257u, 257u, rund::kernel::ComputeDomain::U32,
                        "count257.prefix")) {
    return false;
  }
  return BackendRunsStencilValueCases(pick) && BackendRunsRangeParity(pick) &&
         StencilMatch(
             RuntimeSharedProbeMatchesContract(
                 VulkanMaximumSharedShapeContract(pick, *adapter), "vulkan"),
             "vulkan.maximum-shared-capability");
#else
  return false;
#endif
}

} // namespace node_accel_contract
