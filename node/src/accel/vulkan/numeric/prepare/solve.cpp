#include "../resource.hpp"
#include "../source.hpp"

#include "../../../solve/shape.hpp"

#include <kernel/program/compute/solve/identity.hpp>

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanSolve(const rund::AccelDevice &pick,
                                    const rund::kernel::SolveDesc &desc,
                                    const rund::kernel::SolvePlan &plan,
                                    const SolveBinds &bindings,
                                    const KernelPreparationMode mode,
                                    std::shared_ptr<void> &out) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  out.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || !SolveShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_solve_invalid"};
  }
  const bool wide = plan.element_bytes == 8u;
  if (!wide && plan.element_bytes != 4u) {
    return RejectElement(*adapter, "compute_solve_element_unsupported");
  }
  auto *const raw = new VulkanNumericPrepared{};
  std::shared_ptr<void> owner{raw, DestroyNumericPrepared};
  raw->adapter = adapter;
  const bool aux = plan.input == rund::kernel::SolveInput::Factor &&
                   plan.factor == rund::kernel::FactorOp::LU;
  VulkanResidentReq requests[] = {
      {bindings.primary, bindings.primary_handle, &raw->resident[0]},
      {aux ? bindings.aux : bindings.status,
       aux ? bindings.aux_handle : bindings.status_handle, &raw->resident[1]},
      {bindings.rhs, bindings.rhs_handle, &raw->resident[2]},
      {bindings.output, bindings.output_handle, &raw->resident[3]},
      {bindings.status, bindings.status_handle, &raw->resident[4]},
  };
  rund::AccelCheck check = LookupPrepared(pick, *raw, requests);
  if (!check.ok) {
    return check;
  }
  raw->binding_count = 6u;
  raw->bindings[1] = ResidentBinding(raw->resident[0]);
  raw->bindings[2] = aux ? ResidentBinding(raw->resident[1])
                         : VulkanStorageBindingFor(raw->dummy);
  raw->bindings[3] = ResidentBinding(raw->resident[2]);
  raw->bindings[4] = ResidentBinding(raw->resident[3]);
  raw->bindings[5] = ResidentBinding(raw->resident[4]);
  raw->outputs[0] = raw->resident[3].device_buffer;
  raw->outputs[1] = raw->resident[4].device_buffer;
  raw->output_count = 2u;
  raw->status = raw->resident[4].device_buffer;
  raw->status_binding = ResidentBinding(raw->resident[4]);
  raw->status_count = plan.status_count;
  raw->status_kind = VulkanNumericStatusKind::Solve;
  raw->groups = static_cast<std::uint32_t>(plan.batch_count);
  raw->dispatches = plan.pass_count;
  NumericParams params{static_cast<rund::kernel::u64>(plan.factor),
                       static_cast<rund::kernel::u64>(plan.layout),
                       plan.rows,
                       0u,
                       0u,
                       plan.batch_count,
                       plan.rhs_cols,
                       0u,
                       0u,
                       static_cast<rund::kernel::u32>(plan.input),
                       static_cast<rund::kernel::u32>(plan.pivot)};
  const auto hash = rund::kernel::HashSolve(desc);
  check = FinalizePrepared(
      *raw, params, 6u,
      NumericPseudoPlan(hash, wide ? rund::kernel::ComputeScalar::Lane64
                                   : rund::kernel::ComputeScalar::Lane32),
      wide ? SolveSource64() : SolveSource(), FixedPolicy(plan.fixed_format),
      mode);
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
