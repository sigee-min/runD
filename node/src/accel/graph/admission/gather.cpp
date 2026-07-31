#include <accel/graph/node.hpp>
#include <kernel/program/compute/gather/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/gather/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitGatherNode(const rund::AccelGraphNode &node,
                            GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Gather>();
  active.desc = node.gather;
  active.plan = rund::kernel::PlanGather(active.desc);
  const rund::kernel::GatherHash gather_hash =
      rund::kernel::HashGather(node.gather);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Gather) ||
      !active.plan.ok || !signature.ok ||
      !GatherBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != gather_hash.hi ||
      node.primitive_hash_lo != gather_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_gather_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
