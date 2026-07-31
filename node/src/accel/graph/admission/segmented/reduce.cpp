#include <accel/graph/node.hpp>
#include <kernel/program/compute/segmented/reduce/identity.hpp>

#include "../local.hpp"

#include <kernel/program/compute/segmented/reduce/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitSegmentedReduceNode(const rund::AccelGraphNode &node,
                                     GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::SegmentedReduce>();
  active.desc = node.segmented_reduce;
  active.plan = rund::kernel::PlanSegmentedReduce(active.desc);
  const rund::kernel::SegmentedReduceHash hash =
      rund::kernel::HashSegmentedReduce(node.segmented_reduce);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::SegmentedReduce) ||
      !active.plan.ok || !signature.ok ||
      !SegmentedReduceBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason =
      "accel_kernel_segmented_reduce_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
