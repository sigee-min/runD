#include <accel/context/value.hpp>
#include <accel/kernel/check.hpp>
#include <accel/kernel/value.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] constexpr bool ExecutableApi(const rund::AccelApi api) noexcept {
  return api == rund::AccelApi::Cpu || api == rund::AccelApi::Metal ||
         api == rund::AccelApi::Vulkan;
}

[[nodiscard]] KernelAdmission
ProjectValidatedKernelAdmission(const rund::AccelKernel &kernel) noexcept {
  return KernelAdmission{
      .check = rund::AccelKernelCheck{true, "ok"},
      .kernel_id = kernel.kernel_id,
      .context_id = kernel.context_id,
      .graph_id_hi = kernel.graph_id_hi,
      .graph_id_lo = kernel.graph_id_lo,
      .node_count = kernel.node_count,
      .api = kernel.api,
      .scalar = kernel.scalar,
      .domain = kernel.domain,
      .frozen_caps = kernel.frozen_caps,
      .owner = kernel.owner,
  };
}

} // namespace

std::shared_ptr<KernelToken>
AdmitKernelTokenWithContext(const rund::AccelKernel &kernel,
                            const ContextAdmission &context_admission) {
  if (!kernel.check.ok || !SameReason(kernel.check.reason, "ok") ||
      !SameReason(kernel.reason, "ok") || kernel.owner == nullptr ||
      kernel.kernel_id == 0u ||
      (kernel.graph_id_hi == 0u && kernel.graph_id_lo == 0u) ||
      kernel.node_count == 0u || !ExecutableApi(kernel.api) ||
      !rund::kernel::ComputeScalarValid(kernel.scalar) ||
      !kernel.frozen_caps.ok) {
    return {};
  }

  const std::shared_ptr<KernelToken> token =
      LookupKernelToken(kernel.owner, kernel.kernel_id);
  if (token == nullptr || !SameObject(token, kernel.owner) ||
      !context_admission.check.ok) {
    return {};
  }

  if (kernel.context_id != context_admission.context_id ||
      token->kernel_id != kernel.kernel_id ||
      token->context_id != context_admission.context_id ||
      token->context_id != kernel.context_id ||
      token->graph_id_hi != kernel.graph_id_hi ||
      token->graph_id_lo != kernel.graph_id_lo ||
      token->node_count != kernel.node_count ||
      token->api != context_admission.api || token->api != kernel.api ||
      token->scalar != kernel.scalar || token->domain != kernel.domain ||
      !SameObject(token->context_owner, context_admission.owner) ||
      !SameCaps(token->frozen_caps, context_admission.caps) ||
      !SameCaps(kernel.frozen_caps, context_admission.caps)) {
    return {};
  }

  return token;
}

KernelAdmission AdmitKernelForSupport(const rund::AccelContext &context,
                                      const rund::AccelKernel &kernel) {
  const std::shared_ptr<KernelToken> token =
      AdmitKernelTokenWithContext(kernel, AdmitContextForSupport(context));
  return token == nullptr ? RejectAdmission("accel_kernel_graph_invalid")
                          : ProjectValidatedKernelAdmission(kernel);
}

KernelExecution AdmitKernelForExecution(const rund::AccelContext &context,
                                        const rund::AccelKernel &kernel) {
  const ContextAdmission context_admission = AdmitContextForSupport(context);
  const std::shared_ptr<KernelToken> token =
      AdmitKernelTokenWithContext(kernel, context_admission);
  if (token == nullptr) {
    return KernelExecution{.admission =
                               RejectAdmission("accel_kernel_graph_invalid"),
                           .context_admission = context_admission};
  }

  const KernelAdmission admission = ProjectValidatedKernelAdmission(kernel);
  if (token->graph_roles.size() != token->graph_shapes.size() ||
      token->graph_roles.size() != token->graph_visibilities.size() ||
      token->graph_roles.size() != token->graph_alias_representatives.size() ||
      token->required_barriers.size() != token->steps.size()) {
    return KernelExecution{
        .admission = KernelAdmission{.check = rund::AccelKernelCheck{
                                         false, "accel_kernel_graph_invalid"}}};
  }

  return KernelExecution{
      .admission = admission,
      .context_admission = context_admission,
      .graph_roles = token->graph_roles,
      .graph_shapes = token->graph_shapes,
      .graph_visibilities = token->graph_visibilities,
      .graph_alias_representatives = token->graph_alias_representatives,
      .resets = token->resets,
      .steps = token->steps,
      .required_barriers = token->required_barriers,
      .removed_dispatch_count = token->removed_dispatch_count,
      .original_operation_count = token->original_operation_count,
      .fused_operation_count = token->fused_operation_count,
      .fusion_rejection_count = token->fusion_rejection_count,
      .fusion_reason = token->fusion_reason,
  };
}

} // namespace rund::node::accel::detail
