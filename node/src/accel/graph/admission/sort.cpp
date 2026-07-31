#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <kernel/program/compute/sort/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/sort/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitSortNode(const rund::AccelGraphNode &node,
                          GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Sort>();
  active.desc = SortDescFor(node);
  active.plan = rund::kernel::PlanSort(active.desc);
  const rund::kernel::SortHash sort_hash = rund::kernel::HashSort(active.desc);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Sort) ||
      !active.plan.ok || !signature.ok || !SortBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != sort_hash.hi ||
      node.primitive_hash_lo != sort_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_sort_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
