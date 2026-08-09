#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "src/accel/context/internal/support.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/stencil/metal.hpp"
#include "src/accel/stencil/shape.hpp"
#include "src/accel/stencil/vulkan.hpp"

#include "bindings.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>

namespace node_accel_contract::stencil {
namespace match_detail {

enum class ForcedRangePath : std::uint8_t {
  Direct,
  PrefixDifference,
  BlockPrefixSuffix,
};

[[nodiscard]] inline const char *
BackendLastError(const rund::AccelDevice &pick) {
  if (pick.backend.last_error == nullptr) {
    return "backend_last_error_absent";
  }
  const char *const reason = pick.backend.last_error(pick.backend.context);
  return reason == nullptr ? "backend_last_error_null" : reason;
}

[[nodiscard]] inline bool Fail(const char *const stage,
                               const char *const reason) {
  std::cerr << "stencil match stage failed: " << stage
            << " reason: " << (reason == nullptr ? "null" : reason) << '\n';
  return false;
}

[[nodiscard]] inline std::optional<rund::node::accel::detail::RangeCaps>
ForcedCapabilities(const rund::node::accel::detail::RangeCaps &base,
                   const ForcedRangePath path) noexcept {
  using namespace rund::node::accel::detail;
  std::uint8_t support = RangeSupportBit(RangeSupport::Direct);
  switch (path) {
  case ForcedRangePath::Direct:
    break;
  case ForcedRangePath::PrefixDifference:
    support |= RangeSupportBit(RangeSupport::PrefixDifference);
    break;
  case ForcedRangePath::BlockPrefixSuffix:
    support |= RangeSupportBit(RangeSupport::BlockPrefixSuffix);
    break;
  }
  return RangeCaps::gpu(base.source_variant(), base.legal_width_mask(),
                        base.maximum_threads_per_workgroup(),
                        base.shared_memory_occupancy_budget(),
                        base.shared_memory_limit(), base.maximum_group_count(),
                        support);
}

[[nodiscard]] constexpr rund::node::accel::detail::RangePath
ExpectedCandidate(const ForcedRangePath path) noexcept {
  using namespace rund::node::accel::detail;
  switch (path) {
  case ForcedRangePath::Direct:
    return RangePath::Direct;
  case ForcedRangePath::PrefixDifference:
    return RangePath::PrefixDifference;
  case ForcedRangePath::BlockPrefixSuffix:
    return RangePath::BlockPrefixSuffix;
  }
  return RangePath::Direct;
}

} // namespace match_detail

template <typename T, std::size_t Count>
[[nodiscard]] bool MatchesReference(const rund::AccelDevice &pick,
                                    const rund::kernel::ComputeScalar scalar,
                                    const rund::kernel::ComputeDomain domain,
                                    const rund::kernel::StencilOp op,
                                    const rund::kernel::StencilElement element,
                                    const rund::kernel::u64 radius,
                                    const std::array<T, Count> &input) {
  namespace fix = node_accel_contract::primitive;
  namespace run = node_accel_contract::stencil::match;
  if (!pick.check.ok) {
    return match_detail::Fail("pick", pick.check.reason);
  }
  const bool signed_domain = domain == rund::kernel::ComputeDomain::I32 ||
                             domain == rund::kernel::ComputeDomain::I64 ||
                             domain == rund::kernel::ComputeDomain::Fixed;
  const run::Reference<T, Count> ref =
      run::BuildReference(op, element, radius, input, signed_domain);
  if (!ref.ok) {
    return match_detail::Fail("reference", "reference_failed");
  }
  run::Resources<T, Count> resources =
      run::BuildResources(pick, scalar, domain, op, element, radius, input);
  if (!resources.kernel.check.ok) {
    return match_detail::Fail("compile", resources.kernel.check.reason);
  }
  const auto bindings = run::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return match_detail::Fail("run", match_detail::BackendLastError(pick));
  }
  std::array<T, Count> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, downloaded.data(),
      downloaded.size() * sizeof(T));
  if (!download.ok) {
    return match_detail::Fail("download", download.reason);
  }
  if (fix::HashValues(downloaded.data(), downloaded.size()) !=
      fix::HashValues(ref.expected.data(), ref.expected.size())) {
    return match_detail::Fail("compare", "output_mismatch");
  }
  return true;
}

// This deliberately bypasses graph admission only after it has constructed
// the same validated context and resident bindings. The planner remains the
// sole selection authority: the test narrows its legal support mask to force
// one executable family, then compares each forced backend run to the common
// CPU reference.
template <typename T, std::size_t Count>
[[nodiscard]] bool MatchesForcedPathReference(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element, const rund::kernel::u64 radius,
    const std::array<T, Count> &input,
    const match_detail::ForcedRangePath path) {
  namespace detail = rund::node::accel::detail;
  namespace fix = node_accel_contract::primitive;
  namespace run = node_accel_contract::stencil::match;

  if (!pick.check.ok) {
    return match_detail::Fail("forced.pick", pick.check.reason);
  }
  const bool signed_domain = domain == rund::kernel::ComputeDomain::I32 ||
                             domain == rund::kernel::ComputeDomain::I64 ||
                             domain == rund::kernel::ComputeDomain::Fixed;
  const run::Reference<T, Count> reference =
      run::BuildReference(op, element, radius, input, signed_domain);
  if (!reference.ok) {
    return match_detail::Fail("forced.reference", "reference_failed");
  }
  run::Resources<T, Count> resources =
      run::BuildResources(pick, scalar, domain, op, element, radius, input);
  if (!resources.kernel.check.ok) {
    return match_detail::Fail("forced.compile", resources.kernel.check.reason);
  }
  const detail::ContextAdmission admission =
      detail::AdmitContextForSupport(resources.context);
  std::shared_ptr<void> input_handle;
  std::shared_ptr<void> output_handle;
  if (!admission.check.ok || admission.pick == nullptr ||
      !detail::ValidateAccelBufferForSupport(admission, resources.input,
                                             input_handle)
           .ok ||
      !detail::ValidateAccelBufferForSupport(admission, resources.output,
                                             output_handle)
           .ok) {
    return match_detail::Fail("forced.bind", "resident_binding_invalid");
  }

  const rund::kernel::StencilDesc desc{
      .op = op,
      .element = element,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = input.size(),
      .radius = radius,
  };
  const rund::kernel::StencilPlan semantic = rund::kernel::PlanStencil(desc);
  const std::optional<detail::RangeShape> shape =
      detail::StencilRangeShape(semantic, domain);
  const detail::RangeCaps base =
      admission.pick->raw.api == rund::AccelApi::Metal
          ? detail::MetalRangeCaps(admission.pick->raw)
      : admission.pick->raw.api == rund::AccelApi::Vulkan
          ? detail::VulkanRangeCaps(admission.pick->raw)
          : detail::RangeCaps::unavailable();
  const std::optional<detail::RangeCaps> capabilities =
      match_detail::ForcedCapabilities(base, path);
  const detail::RangePlan range =
      shape.has_value() && capabilities.has_value()
          ? detail::PlanRange(*shape, *capabilities)
          : detail::RangePlan::rejected(
                "compute_range_aggregate_candidate_unavailable");
  if (!semantic.ok || !range.ok() ||
      range.candidate().disposition() !=
          match_detail::ExpectedCandidate(path)) {
    return match_detail::Fail("forced.plan", range.reason());
  }

  const detail::StencilBinds bindings{
      .input = &resources.input.resident,
      .input_handle = &input_handle,
      .output = &resources.output.resident,
      .output_handle = &output_handle,
  };
  const rund::AccelCheck executed =
      admission.pick->raw.api == rund::AccelApi::Metal
          ? detail::ExecuteMetalStencil(admission.pick->raw, desc, semantic,
                                        domain, bindings, range)
      : admission.pick->raw.api == rund::AccelApi::Vulkan
          ? detail::ExecuteVulkanStencil(admission.pick->raw, desc, semantic,
                                         domain, bindings, range)
          : rund::AccelCheck{false, "compute_adapter_unavailable"};
  if (!executed.ok) {
    return match_detail::Fail("forced.execute", executed.reason);
  }
  std::array<T, Count> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, downloaded.data(),
      downloaded.size() * sizeof(T));
  if (!download.ok) {
    return match_detail::Fail("forced.download", download.reason);
  }
  return fix::HashValues(downloaded.data(), downloaded.size()) ==
         fix::HashValues(reference.expected.data(), reference.expected.size());
}

} // namespace node_accel_contract::stencil
