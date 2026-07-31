#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <kernel/program/compute/scatter/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/scatter/plan.hpp>
#include <kernel/program/compute/scatter/reduce/identity.hpp>
#include <kernel/program/compute/scatter/reduce/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitScatterNode(const rund::AccelGraphNode &node,
                             GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Scatter>();
  active.desc = ScatterDescFor(node);
  active.plan = rund::kernel::PlanScatter(active.desc);
  const rund::kernel::ScatterHash scatter_hash =
      rund::kernel::HashScatter(active.desc);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Scatter) ||
      !active.plan.ok || !signature.ok ||
      !ScatterBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != scatter_hash.hi ||
      node.primitive_hash_lo != scatter_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_scatter_backend_unsupported";
  return "ok";
}

const char *AdmitScatterReduceNode(const rund::AccelGraphNode &node,
                                   GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::ScatterReduce>();
  active.desc = node.scatter_reduce;
  active.plan = rund::kernel::PlanScatterReduce(active.desc);
  const auto hash = rund::kernel::HashScatterReduce(active.desc);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::ScatterReduce) ||
      !active.plan.ok || !signature.ok ||
      !ScatterReduceBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_scatter_reduce_backend";
  return "ok";
}

} // namespace rund::node::accel::detail
