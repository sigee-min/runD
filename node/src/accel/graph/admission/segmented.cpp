#include <accel/graph/node.hpp>
#include <kernel/program/compute/segmented/scan/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/segmented/scan/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitSegmentedScanNode(const rund::AccelGraphNode &node,
                                   GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::SegmentedScan>();
  active.desc = node.segmented_scan;
  active.plan = rund::kernel::PlanSegmentedScan(active.desc);
  const rund::kernel::SegmentedScanHash hash =
      rund::kernel::HashSegmentedScan(node.segmented_scan);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::SegmentedScan) ||
      !active.plan.ok || !signature.ok ||
      !SegmentedScanBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason =
      "accel_kernel_segmented_scan_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
