#include "../resource.hpp"
#include "../source.hpp"

#include "../../../transform/shape.hpp"

#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/stage.hpp>

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanTransform(const rund::AccelDevice &pick,
                                        const rund::kernel::TransformDesc &desc,
                                        const rund::kernel::TransformPlan &plan,
                                        const TransformBinds &bindings,
                                        const KernelPreparationMode mode,
                                        std::shared_ptr<void> &out) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  out.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || !TransformShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_transform_invalid"};
  }
  const bool wide = bindings.input_real->element_bytes == 8u;
  if (!wide && bindings.input_real->element_bytes != 4u) {
    return RejectElement(*adapter, "compute_transform_element_unsupported");
  }
  const auto local =
      rund::kernel::transform_stage::Describe(plan.element_count, 1u);
  const rund::kernel::u64 groups = rund::kernel::transform_stage::Groups(
      rund::kernel::transform_stage::Threads(plan.element_count, local));
  if (groups == 0u || groups > adapter->max_dispatch_groups) {
    return rund::AccelCheck{false, "compute_transform_dispatch_unsupported"};
  }
  auto *const raw = new VulkanNumericPrepared{};
  std::shared_ptr<void> owner{raw, DestroyNumericPrepared};
  raw->adapter = adapter;
  VulkanResidentReq requests[] = {
      {bindings.input_real, bindings.input_real_handle, &raw->resident[0]},
      {bindings.input_imag, bindings.input_imag_handle, &raw->resident[1]},
      {bindings.output_real, bindings.output_real_handle, &raw->resident[2]},
      {bindings.output_imag, bindings.output_imag_handle, &raw->resident[3]},
  };
  rund::AccelCheck check = LookupPrepared(pick, *raw, requests);
  if (!check.ok) {
    return check;
  }
  raw->binding_count = 6u;
  raw->bindings[1] = ResidentBinding(raw->resident[0]);
  raw->bindings[2] = ResidentBinding(raw->resident[1]);
  raw->bindings[3] = ResidentBinding(raw->resident[2]);
  raw->bindings[4] = ResidentBinding(raw->resident[3]);
  raw->bindings[5] = VulkanStorageBindingFor(
      plan.workspace_bytes == 0u ? raw->dummy : raw->twiddle);
  raw->outputs[0] = raw->resident[2].device_buffer;
  raw->outputs[1] = raw->resident[3].device_buffer;
  raw->output_count = 2u;
  raw->groups = static_cast<std::uint32_t>(groups);
  raw->transform_count = plan.element_count;
  raw->dispatches = plan.pass_count;
  NumericParams params{static_cast<rund::kernel::u64>(plan.op),
                       1u,
                       plan.element_count,
                       0u,
                       0u,
                       plan.normalization_divisor,
                       0u,
                       0u,
                       0u,
                       static_cast<rund::kernel::u32>(plan.direction),
                       static_cast<rund::kernel::u32>(plan.normalization)};
  const auto hash = rund::kernel::HashTransform(desc);
  if (!PrepareTwiddle(*raw, plan)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  check = FinalizePrepared(
      *raw, params, 6u,
      NumericPseudoPlan(hash, wide ? rund::kernel::ComputeScalar::Lane64
                                   : rund::kernel::ComputeScalar::Lane32),
      wide ? TransformSource64() : TransformSource(),
      FixedPolicy(plan.fixed_format), mode,
      sizeof(rund::kernel::transform_stage::Batch));
  if (!check.ok) {
    return check;
  }
  out = std::move(owner);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)mode;
  (void)out;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
