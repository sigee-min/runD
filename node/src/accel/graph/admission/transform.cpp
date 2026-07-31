#include <accel/graph/node.hpp>
#include <kernel/program/compute/transform/identity.hpp>

#include "local.hpp"

#include <kernel/program/compute/transform/plan.hpp>

namespace rund::node::accel::detail {

const char *AdmitTransformNode(const rund::AccelGraphNode &node,
                               GraphCompileNode &compile_data) {
  auto &active = compile_data.operation.set<operation::Transform>();
  active.desc = node.transform;
  active.plan = rund::kernel::PlanTransform(active.desc);
  const rund::kernel::TransformHash transform_hash =
      rund::kernel::HashTransform(node.transform);
  const AdmissionSignature signature = AdmitSignature(node, active.plan);
  if (!PrimitivePayloadOnly(node, rund::kernel::NodeKind::Transform) ||
      !active.plan.ok || !signature.ok ||
      !TransformBindingsOk(node, active.plan) ||
      node.element_count != active.plan.element_count ||
      node.primitive_hash_hi != transform_hash.hi ||
      node.primitive_hash_lo != transform_hash.lo) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  compile_data.artifact.reason = "ok";
  return "ok";
}

} // namespace rund::node::accel::detail
