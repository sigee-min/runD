#include <accel/graph/node.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

void AppendKernelGraphNode(
    const rund::AccelGraphNode &node, const GraphCompileNode &compile_data,
    const std::vector<rund::kernel::GraphBufferRef> &buffers,
    GraphCompileState &state) {
  if (node.barrier_before && !state.fusion_nodes.empty()) {
    state.fusion_nodes.back().writes_visible = true;
  }
  state.fusion_nodes.push_back(rund::kernel::FusionNodePolicy{
      .direct_read_mask = compile_data.fusion_supported
                              ? compile_data.map_metadata.direct_read_mask
                              : 0u,
      .supported = compile_data.fusion_supported,
      .writes_visible = compile_data.fusion_write_visible,
      .binding_count = compile_data.fusion_supported
                           ? static_cast<rund::kernel::u32>(
                                 compile_data.cpu_input.parsed.bindings.size())
                           : 0u,
      .ir_node_count = compile_data.fusion_supported
                           ? static_cast<rund::kernel::u32>(
                                 compile_data.cpu_input.parsed.nodes.size())
                           : 0u});
  state.required_barriers.push_back(node.barrier_before ? 1u : 0u);
  if (node.kind == rund::kernel::NodeKind::Map) {
    state.node_storage.push_back(rund::kernel::GraphNode{
        .op_hash_hi = node.ir->op_hash_hi,
        .op_hash_lo = node.ir->op_hash_lo,
        .buffers = buffers.empty() ? nullptr : buffers.data(),
        .buffer_count = static_cast<rund::kernel::u64>(buffers.size()),
        .kind = node.kind,
        .element_count = compile_data.element_count,
        .control = compile_data.control});
    return;
  }
  state.node_storage.push_back(rund::kernel::GraphNode{
      .buffers = buffers.empty() ? nullptr : buffers.data(),
      .buffer_count = static_cast<rund::kernel::u64>(buffers.size()),
      .kind = node.kind,
      .primitive_hash_hi = compile_data.primitive_hash_hi,
      .primitive_hash_lo = compile_data.primitive_hash_lo,
      .element_count = compile_data.element_count,
      .control = compile_data.control});
}

} // namespace rund::node::accel::detail
