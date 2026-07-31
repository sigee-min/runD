#include <accel/graph/node.hpp>
#include <kernel/program/compute/stencil/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitStencilNode(const rund::AccelGraphNode &node,
                             GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Stencil>();
  active.desc = node.stencil;
  active.plan = rund::kernel::PlanStencil(active.desc);
  const rund::kernel::StencilHash hash =
      rund::kernel::HashStencil(node.stencil);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Stencil) ||
      !active.plan.ok || !signature.ok ||
      !StencilBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != hash.hi || node.primitive_hash_lo != hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "accel_kernel_stencil_backend_unsupported";
  return "ok";
}

} // namespace rund::node::accel::detail
