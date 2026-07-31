#include <accel/graph/node.hpp>
#include <kernel/program/compute/matrix/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/matrix/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitMatrixNode(const rund::AccelGraphNode &node,
                            GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Matrix>();
  active.desc = node.matrix;
  active.plan = rund::kernel::PlanMatrix(active.desc);
  const rund::kernel::MatrixHash matrix_hash =
      rund::kernel::HashMatrix(node.matrix);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Matrix) ||
      !active.plan.ok || !signature.ok ||
      !MatrixBindingsOk(node, active.plan) ||
      node.element_count != active.plan.output_count ||
      node.primitive_hash_hi != matrix_hash.hi ||
      node.primitive_hash_lo != matrix_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "ok";
  return "ok";
}

} // namespace rund::node::accel::detail
