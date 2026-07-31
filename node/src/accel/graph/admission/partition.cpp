#include <accel/graph/node.hpp>
#include <kernel/program/compute/partition/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/partition/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitPartitionNode(const rund::AccelGraphNode &node,
                               GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Partition>();
  active.desc = node.partition;
  active.plan = rund::kernel::PlanPartition(active.desc);
  const rund::kernel::PartitionHash partition_hash =
      rund::kernel::HashPartition(node.partition);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Partition) ||
      !active.plan.ok || !signature.ok ||
      !PartitionBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != partition_hash.hi ||
      node.primitive_hash_lo != partition_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_partition_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
