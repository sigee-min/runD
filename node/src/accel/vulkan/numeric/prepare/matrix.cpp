#include "../resource.hpp"
#include "../source.hpp"
#include "../../kernel/pipeline/template.hpp"

#include "../../../matrix/shape.hpp"

#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/matrix/tile.hpp>

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanMatrix(const rund::AccelDevice &pick,
                                     const rund::kernel::MatrixDesc &desc,
                                     const rund::kernel::MatrixPlan &plan,
                                     const MatrixBinds &bindings,
                                     const KernelPreparationMode mode,
                                     std::shared_ptr<void> &out,
                                     const VulkanKernelImmutablePipelines
                                         *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  out.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || !MatrixShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_matrix_invalid"};
  }
  const bool wide = plan.element_bytes == 8u;
  if (!wide && plan.element_bytes != 4u) {
    return RejectElement(*adapter, "compute_matrix_element_unsupported");
  }
  const rund::kernel::matrix_tile::Tiles tiles =
      rund::kernel::matrix_tile::Count(plan.rows, plan.cols, plan.batch_count);
  if (!tiles.ok || tiles.count > adapter->max_dispatch_groups) {
    return rund::AccelCheck{false, "compute_matrix_dispatch_unsupported"};
  }
  auto *const raw = new VulkanNumericPrepared{};
  std::shared_ptr<void> owner{raw, DestroyNumericPrepared};
  raw->adapter = adapter;
  const bool transpose = plan.op == rund::kernel::MatrixOp::Transpose;
  VulkanResidentReq requests[] = {
      {bindings.left, bindings.left_handle, &raw->resident[0]},
      {transpose ? bindings.output : bindings.right,
       transpose ? bindings.output_handle : bindings.right_handle,
       &raw->resident[1]},
      {bindings.output, bindings.output_handle, &raw->resident[2]},
  };
  rund::AccelCheck check = LookupPrepared(pick, *raw, requests);
  if (!check.ok) {
    return check;
  }
  raw->binding_count = 4u;
  raw->bindings[1] = ResidentBinding(raw->resident[0]);
  raw->bindings[2] = transpose ? VulkanStorageBindingFor(raw->dummy)
                               : ResidentBinding(raw->resident[1]);
  raw->bindings[3] = ResidentBinding(raw->resident[2]);
  raw->outputs[0] = raw->resident[2].device_buffer;
  raw->output_count = 1u;
  raw->groups = static_cast<std::uint32_t>(tiles.count);
  raw->dispatches = plan.pass_count;
  NumericParams params{static_cast<rund::kernel::u64>(plan.op),
                       static_cast<rund::kernel::u64>(plan.layout),
                       plan.rows,
                       plan.cols,
                       plan.inner,
                       plan.batch_count};
  params.mode = static_cast<rund::kernel::u32>(plan.arithmetic);
  const auto hash = rund::kernel::HashMatrix(
      rund::kernel::MatrixDesc{.element_bytes = desc.element_bytes});
  const rund::kernel::ComputeDomain executable_domain =
      plan.arithmetic == rund::kernel::MatrixArithmetic::Fixed
          ? rund::kernel::ComputeDomain::Fixed
          : (plan.arithmetic == rund::kernel::MatrixArithmetic::SignedWrap
                 ? (wide ? rund::kernel::ComputeDomain::I64
                         : rund::kernel::ComputeDomain::I32)
                 : (wide ? rund::kernel::ComputeDomain::U64
                         : rund::kernel::ComputeDomain::U32));
  VulkanCollectivePipeline *const pipeline =
      pipelines == nullptr
          ? AcquireNumericPipeline(
                *adapter, 4u, 0u,
                NumericPseudoPlan(
                    hash,
                    wide ? rund::kernel::ComputeScalar::Lane64
                         : rund::kernel::ComputeScalar::Lane32,
                    executable_domain,
                    plan.arithmetic == rund::kernel::MatrixArithmetic::Fixed
                        ? plan.fixed_format
                        : rund::kernel::ComputeFixedFormat{}),
                wide ? MatrixSource64() : MatrixSource(),
                MatrixPolicy(plan.arithmetic, plan.fixed_format))
          : pipelines->borrow(rund::kernel::NodeKind::Matrix, 1u, 0u, 4u, 1u);
  check = FinalizePrepared(*raw, params, 4u, pipeline, mode);
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
