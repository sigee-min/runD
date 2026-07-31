#include <accel/graph/node.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/spectrum/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitSpectrumNode(const rund::AccelGraphNode &node,
                              GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Spectrum>();
  active.desc = node.spectrum;
  active.plan = rund::kernel::PlanSpectrum(active.desc);
  const rund::kernel::SpectrumHash hash =
      rund::kernel::HashSpectrum(node.spectrum);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Spectrum) ||
      !active.plan.ok || !signature.ok ||
      !SpectrumBindingsOk(node, active.plan) ||
      node.element_count != active.plan.value_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "ok";
  return "ok";
}

} // namespace rund::node::accel::detail
