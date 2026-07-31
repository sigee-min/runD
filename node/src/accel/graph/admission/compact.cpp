#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <kernel/program/compute/compact/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/compact/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitCompactNode(const rund::AccelGraphNode &node,
                             GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Compact>();
  active.desc = CompactDescFor(node);
  active.plan = rund::kernel::PlanCompact(active.desc);
  const rund::kernel::CompactHash compact_hash =
      rund::kernel::HashCompact(active.desc);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Compact) ||
      !active.plan.ok || !signature.ok ||
      !CompactBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != compact_hash.hi ||
      node.primitive_hash_lo != compact_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_compact_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
