#include <accel/graph/node.hpp>
#include <kernel/program/compute/stencil/identity.hpp>

#include "../../context/internal/admission.hpp"
#include "../../range_aggregate/plan.hpp"
#include "local.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitStencilNode(const rund::AccelGraphNode &node,
                             const ContextAdmission &admission,
                             const rund::kernel::ComputeDomain domain,
                             GraphCompileNode &compile_data) {
  const rund::kernel::StencilDesc desc = node.stencil;
  const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  const rund::kernel::StencilHash hash =
      rund::kernel::HashStencil(node.stencil);
  const AdmissionSignature signature = AdmitSignature(node, plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Stencil) ||
      !plan.ok || !signature.ok || !StencilBindingsOk(node, plan) ||
      node.element_count != plan.element_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }

  const std::optional<RangeAggregateShape> shape =
      RangeAggregateShape::from_stencil(plan, domain);
  if (!shape.has_value()) {
    return "accel_kernel_graph_invalid";
  }
  if (!admission.check.ok || admission.pick == nullptr ||
      admission.pick->ops == nullptr ||
      admission.pick->ops->range_aggregate_capabilities == nullptr) {
    return "accel_kernel_stencil_backend_unsupported";
  }
  const RangeAggregateCapabilities capabilities =
      admission.pick->ops->range_aggregate_capabilities(admission.pick->raw);
  const RangeAggregatePlan range = PlanRangeAggregate(*shape, capabilities);
  if (!range.ok()) {
    return "accel_kernel_stencil_backend_unsupported";
  }

  compile_data.operation.set<operation::Stencil>(desc, plan, range);
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_stencil_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
