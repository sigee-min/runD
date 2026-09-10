#include "local.hpp"

#include "../contract.hpp"
#include "../match/resources.hpp"
#include "src/accel/context/internal/support.hpp"
#include "src/accel/kernel/bindings/range.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/metal/pipeline/cache.hpp"
#include "src/accel/metal/pipeline/template.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/stencil/metal.hpp"
#include "src/accel/stencil/range.hpp"

#include <accel/api.hpp>
#include <accel/runtime.hpp>
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace node_accel_contract::stencil::backend {
namespace {

#if defined(__APPLE__)
[[nodiscard]] rund::node::accel::detail::RangePlan
MetalStencilRangePlan(const rund::AccelDevice &pick,
                      const rund::kernel::StencilPlan &plan,
                      const rund::kernel::ComputeDomain domain,
                      const bool shared_probe = false) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = ProjectStencilRange(plan, domain);
  const RangeCaps base = MetalRangeCaps(pick);
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

[[nodiscard]] RuntimeSharedProbe
MetalMaximumSharedShapeContract(const rund::AccelDevice &pick) {
  using namespace rund::node::accel::detail;
  std::array<rund::kernel::u32, 515u> input{};
  auto fixture = ::node_accel_contract::stencil::match::BuildResources(
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
          admission.pick->raw, plan, rund::kernel::ComputeDomain::U32, true);
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
    const RangePlan selected_range =
        MetalStencilRangePlan(admission.pick->raw, selected_plan,
                              rund::kernel::ComputeDomain::U32, true);
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
  if (!::node_accel_contract::stencil::MatchesCapabilitySharedBoundaryU32(
          pick, maximum->radius_capacity())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}
#endif

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

} // namespace

bool RunMetalRequired() {
  const rund::AccelDevice pick = rund::node::accel::PickAccel(
      ::node_accel_contract::primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return ::node_accel_contract::primitive::PickUnavailableReasonIsPrecise(
        pick, rund::AccelApi::Metal);
  }
  if (!StencilMatch(pick.api == rund::AccelApi::Metal, "pick.api") ||
      !StencilMatch(::node_accel_contract::stencil::BindingsContract(),
                    "bindings")) {
    return false;
  }
  auto *const adapter = static_cast<rund::node::accel::detail::MetalAdapter *>(
      pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    std::cerr << "metal stencil adapter unavailable after successful pick\n";
    return false;
  }
  MetalStencilVariantCacheContract variants{*adapter};
  if (!StencilMatch(::node_accel_contract::stencil::MatchesU32(pick),
                    "sum.u32") ||
      !variants.Observe(pick, "count6.radius1") ||
      !StencilMatch(::node_accel_contract::stencil::MatchesSumI32(pick),
                    "sum.i32") ||
      !variants.Observe(pick, "count6.radius1.signed-sum") ||
      !StencilMatch(::node_accel_contract::stencil::MatchesWideWindowU32(pick),
                    "sum.u32.radius2") ||
      !variants.Observe(pick, "count6.radius2") ||
      !StencilMatch(::node_accel_contract::stencil::MatchesCount65U32(pick),
                    "sum.u32.count65") ||
      !variants.Observe(pick, "count65.radius1") ||
      !StencilMatch(
          ::node_accel_contract::stencil::MatchesRadius64BoundaryU32(pick),
          "sum.u32.count259.radius64") ||
      !variants.Observe(pick, "count259.radius64") ||
      !StencilMatch(
          ::node_accel_contract::stencil::MatchesPrefixDifferenceU32(pick),
          "sum.u32.prefix-difference") ||
      !variants.Observe(pick, "count257.prefix") || !variants.complete()) {
    return false;
  }
#if defined(__APPLE__)
  return RunValue(pick) && RunRange(pick) &&
         StencilMatch(RuntimeSharedProbeMatchesContract(
                          MetalMaximumSharedShapeContract(pick), "metal"),
                      "metal.maximum-shared-capability");
#else
  return false;
#endif
}

} // namespace node_accel_contract::stencil::backend
