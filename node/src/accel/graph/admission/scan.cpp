#include <accel/graph/node.hpp>
#include <kernel/program/compute/scan/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitScanNode(const rund::AccelGraphNode &node,
                          GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Scan>();
  active.desc = node.scan;
  active.plan = rund::kernel::PlanScan(active.desc);
  const rund::kernel::ScanHash scan_hash = rund::kernel::HashScan(node.scan);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Scan) ||
      !active.plan.ok || !signature.ok || !ScanBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != scan_hash.hi ||
      node.primitive_hash_lo != scan_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_scan_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
