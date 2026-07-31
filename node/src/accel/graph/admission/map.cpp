#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include "local.hpp"

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/metadata.hpp>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::kernel::compute_lowering_detail::ComputeInputAdmission
AdmitNode(const rund::AccelGraphNode &node,
          const rund::kernel::ComputeApi api) {
  if (node.ir == nullptr ||
      !rund::kernel::ComputeScalarValid(node.ir->scalar)) {
    return rund::kernel::compute_lowering_detail::ComputeInputAdmission{
        .reason = "accel_kernel_graph_invalid"};
  }
  return rund::kernel::compute_lowering_detail::AdmitComputeInput(*node.ir,
                                                                  api);
}

} // namespace

const char *AdmitMapNode(const rund::AccelGraphNode &node,
                         const rund::kernel::ComputeApi api,
                         GraphCompileNode &compile_data) {
  if (node.primitive_hash_hi != 0u || node.primitive_hash_lo != 0u ||
      node.element_count == 0u ||
      !PrimitivePayloadOnly(node, rund::kernel::NodeKind::Map)) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.cpu_input = AdmitNode(node, api);
  if (!compile_data.cpu_input.ok) {
    return SameReason(compile_data.cpu_input.reason,
                      "accel_kernel_graph_invalid")
               ? "accel_kernel_graph_invalid"
               : "accel_kernel_artifact_invalid";
  }
  if (compile_data.cpu_input.parse_count != 1u) {
    return "accel_kernel_artifact_invalid";
  }
  compile_data.map_metadata =
      rund::kernel::compute_lowering_detail::MetadataFromParsed(
          *node.ir, api, compile_data.cpu_input.parsed);
  const rund::kernel::ExecutionMetadata &run_metadata =
      compile_data.map_metadata;
  if (!run_metadata.ok) {
    return run_metadata.reason;
  }
  const AdmissionSignature signature =
      AdmitSignature(node, rund::kernel::GraphSignatureFor(run_metadata));
  if (!signature.ok) {
    return "accel_kernel_graph_invalid";
  }
  if (!BindingOrderOk(node, run_metadata)) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.signature = signature.signature;
  return "ok";
}

} // namespace rund::node::accel::detail
