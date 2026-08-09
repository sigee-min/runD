#include <accel/graph/node.hpp>
#include <kernel/program/compute/window/identity.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include "../../context/internal/admission.hpp"
#include "../../range_aggregate/plan.hpp"
#include "../../window/shape.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

const char *AdmitWindowNode(const rund::AccelGraphNode &node,
                            const ContextAdmission &admission,
                            GraphCompileNode &compile_data) {
  const rund::kernel::WindowDesc desc = node.window;
  const rund::kernel::WindowPlan plan = rund::kernel::PlanWindow(desc);
  const rund::kernel::WindowHash hash = rund::kernel::HashWindow(desc);
  const AdmissionSignature signature = AdmitSignature(node, plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Window) || !plan.ok ||
      !signature.ok || !WindowBindingsOk(node, plan) ||
      node.element_count != plan.output_count || plan.input_count == 0u ||
      plan.output_count == 0u || node.primitive_hash_hi != hash.hi ||
      node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }

  const std::optional<RangeShape> shape = WindowRangeShape(plan);
  if (!shape.has_value()) {
    return "accel_kernel_graph_invalid";
  }
  if (!admission.check.ok || admission.pick == nullptr ||
      admission.pick->ops == nullptr ||
      admission.pick->ops->range_caps == nullptr) {
    return "accel_kernel_window_backend_unsupported";
  }
  const RangeCaps capabilities =
      admission.pick->ops->range_caps(admission.pick->raw);
  const RangePlan range = PlanRange(*shape, capabilities);
  if (!range.ok()) {
    return "accel_kernel_window_backend_unsupported";
  }

  compile_data.operation.set<operation::Window>(desc, plan, range);
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_window_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
