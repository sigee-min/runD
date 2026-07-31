#include <accel/graph/node.hpp>
#include <kernel/program/compute/histogram/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/histogram/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitHistogramNode(const rund::AccelGraphNode &node,
                               GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Histogram>();
  active.desc = node.histogram;
  active.plan = rund::kernel::PlanHistogram(active.desc);
  const rund::kernel::HistogramHash hash =
      rund::kernel::HashHistogram(node.histogram);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Histogram) ||
      !active.plan.ok || !signature.ok ||
      !HistogramBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_histogram_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
