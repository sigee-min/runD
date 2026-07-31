#include <accel/context/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>

#include "local.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

const char *AppendGraphCompileNode(const ContextAdmission &admission,
                                   const rund::AccelGraph &graph,
                                   const std::uint64_t node_index,
                                   GraphCompileState &state) {
  const rund::AccelGraphNode &node = graph.nodes[node_index];
  if (!rund::kernel::NodeKindValid(node.kind) ||
      !NodeBuffersCanBeWalked(node) ||
      node_index >= std::numeric_limits<std::uint32_t>::max()) {
    return "accel_kernel_graph_invalid";
  }
  if (node.kind == rund::kernel::NodeKind::Map && node.ir == nullptr) {
    return "accel_kernel_graph_invalid";
  }

  GraphCompileNode compile_data{};
  compile_data.fusion_supported =
      node.kind == rund::kernel::NodeKind::Map && !node.control.has_count() &&
      !node.control.has_predicate() && node.ir->scalar == graph.scalar &&
      node.ir->domain == graph.domain &&
      (graph.domain != rund::kernel::ComputeDomain::Fixed ||
       node.ir->fixed_format == graph.fixed_format);
  PopulateBaseCompileData(node, compile_data);
  const char *const primitive_reason =
      AdmitGraphNodePrimitive(node, admission.caps.api, compile_data);
  if (!SameReason(primitive_reason, "ok")) {
    return primitive_reason;
  }

  std::vector<rund::kernel::GraphBufferRef> &buffers =
      state.buffer_storage.emplace_back();
  buffers.reserve(static_cast<std::size_t>(node.buffer_count));
  const char *const buffer_reason = AppendGraphBufferRefs(
      admission, node, SourceStep{static_cast<std::uint32_t>(node_index)},
      state, compile_data, buffers);
  if (!SameReason(buffer_reason, "ok")) {
    return buffer_reason;
  }

  AppendKernelGraphNode(node, compile_data, buffers, state);
  state.compile_nodes.push_back(std::move(compile_data));
  return "ok";
}

rund::kernel::Graph
KernelGraphFor(const GraphCompileState &state,
               const rund::kernel::ComputeScalar scalar,
               const rund::kernel::ComputeDomain domain,
               const rund::kernel::ComputeFixedFormat fixed_format,
               const std::uint64_t *const outputs,
               const std::uint64_t output_count) noexcept {
  return rund::kernel::Graph{
      .nodes = state.node_storage.empty() ? nullptr : state.node_storage.data(),
      .node_count = static_cast<rund::kernel::u64>(state.node_storage.size()),
      .outputs = outputs,
      .output_count = output_count,
      .scalar = scalar,
      .domain = domain,
      .fixed_format = fixed_format,
  };
}

} // namespace rund::node::accel::detail
