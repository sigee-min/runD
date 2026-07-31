#include <accel/graph/node.hpp>
#include <kernel/program/compute/factor/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/factor/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitFactorNode(const rund::AccelGraphNode &node,
                            GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Factor>();
  active.desc = node.factor;
  active.plan = rund::kernel::PlanFactor(active.desc);
  const rund::kernel::FactorHash hash = rund::kernel::HashFactor(node.factor);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Factor) ||
      !active.plan.ok || !signature.ok ||
      !FactorBindingsOk(node, active.plan) ||
      node.element_count != active.plan.factor_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "ok";
  return "ok";
}

} // namespace rund::node::accel::detail
