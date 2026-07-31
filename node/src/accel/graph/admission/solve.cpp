#include <accel/graph/node.hpp>
#include <kernel/program/compute/solve/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/solve/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitSolveNode(const rund::AccelGraphNode &node,
                           GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Solve>();
  active.desc = node.solve;
  active.plan = rund::kernel::PlanSolve(active.desc);
  const rund::kernel::SolveHash hash = rund::kernel::HashSolve(node.solve);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Solve) ||
      !active.plan.ok || !signature.ok || !SolveBindingsOk(node, active.plan) ||
      node.element_count != active.plan.output_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "ok";
  return "ok";
}

} // namespace rund::node::accel::detail
