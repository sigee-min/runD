#include "../resource.hpp"
#include "../source.hpp"
#include "../../kernel/pipeline/template.hpp"

#include "../../../factor/shape.hpp"

#include <kernel/program/compute/factor/identity.hpp>

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanFactor(const rund::AccelDevice &pick,
                                     const rund::kernel::FactorDesc &desc,
                                     const rund::kernel::FactorPlan &plan,
                                     const FactorBinds &bindings,
                                     const KernelPreparationMode mode,
                                     std::shared_ptr<void> &out,
                                     const VulkanKernelImmutablePipelines
                                         *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  out.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || !FactorShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_factor_invalid"};
  }
  const bool wide = plan.element_bytes == 8u;
  if (!wide && plan.element_bytes != 4u) {
    return RejectElement(*adapter, "compute_factor_element_unsupported");
  }
  auto *const raw = new VulkanNumericPrepared{};
  std::shared_ptr<void> owner{raw, DestroyNumericPrepared};
  raw->adapter = adapter;
  const bool aux = plan.op == rund::kernel::FactorOp::LU;
  VulkanResidentReq requests[] = {
      {bindings.input, bindings.input_handle, &raw->resident[0]},
      {bindings.factor, bindings.factor_handle, &raw->resident[1]},
      {aux ? bindings.aux : bindings.status,
       aux ? bindings.aux_handle : bindings.status_handle, &raw->resident[2]},
      {bindings.status, bindings.status_handle, &raw->resident[3]},
  };
  rund::AccelCheck check = LookupPrepared(pick, *raw, requests);
  if (!check.ok) {
    return check;
  }
  raw->binding_count = 5u;
  raw->bindings[1] = ResidentBinding(raw->resident[0]);
  raw->bindings[2] = ResidentBinding(raw->resident[1]);
  raw->bindings[3] = aux ? ResidentBinding(raw->resident[2])
                         : VulkanStorageBindingFor(raw->dummy);
  raw->bindings[4] = ResidentBinding(raw->resident[3]);
  raw->outputs[0] = raw->resident[1].device_buffer;
  raw->outputs[1] = aux ? raw->resident[2].device_buffer : &raw->dummy;
  raw->outputs[2] = raw->resident[3].device_buffer;
  raw->output_count = 3u;
  raw->status = raw->resident[3].device_buffer;
  raw->status_binding = ResidentBinding(raw->resident[3]);
  raw->status_count = plan.status_count;
  raw->status_kind = VulkanNumericStatusKind::Factor;
  raw->groups = static_cast<std::uint32_t>(plan.batch_count);
  raw->dispatches = plan.pass_count;
  NumericParams params{static_cast<rund::kernel::u64>(plan.op),
                       static_cast<rund::kernel::u64>(plan.layout),
                       plan.rows,
                       plan.cols,
                       0u,
                       plan.batch_count,
                       0u,
                       plan.factor_count / plan.batch_count,
                       0u,
                       static_cast<rund::kernel::u32>(plan.output),
                       static_cast<rund::kernel::u32>(plan.pivot)};
  const auto hash = rund::kernel::HashFactor(
      rund::kernel::FactorDesc{.element_bytes = desc.element_bytes});
  VulkanCollectivePipeline *const pipeline =
      pipelines == nullptr
          ? AcquireNumericPipeline(
                *adapter, 5u, 0u,
                NumericPseudoPlan(
                    hash,
                    wide ? rund::kernel::ComputeScalar::Lane64
                         : rund::kernel::ComputeScalar::Lane32,
                    rund::kernel::ComputeDomain::Fixed, plan.fixed_format),
                wide ? FactorSource64() : FactorSource(),
                FixedPolicy(plan.fixed_format))
          : pipelines->borrow(rund::kernel::NodeKind::Factor, 1u, 0u, 5u, 1u);
  check = FinalizePrepared(*raw, params, 5u, pipeline, mode);
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
  (void)pipelines;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
