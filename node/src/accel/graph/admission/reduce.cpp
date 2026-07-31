#include <accel/graph/node.hpp>
#include <kernel/program/compute/reduce/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/reduce/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitReduceNode(const rund::AccelGraphNode &node,
                            GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Reduce>();
  active.desc = node.reduce;
  active.plan = rund::kernel::PlanReduce(active.desc);
  const rund::kernel::ReduceHash reduce_hash =
      rund::kernel::HashReduce(node.reduce);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Reduce) ||
      !active.plan.ok || !signature.ok ||
      !ReduceBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != reduce_hash.hi ||
      node.primitive_hash_lo != reduce_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_reduce_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
