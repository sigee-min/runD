#include <accel/graph/node.hpp>

#include "admission/local.hpp"
namespace rund::node::accel::detail {

const char *AdmitGraphNodePrimitive(const rund::AccelGraphNode &node,
                                    const rund::kernel::ComputeApi api,
                                    GraphCompileNode &compile_data) {
  if (node.kind == rund::kernel::NodeKind::Map) {
    return AdmitMapNode(node, api, compile_data);
  }
  if (ScanGraphNode(node.kind)) {
    return AdmitScanNode(node, compile_data);
  }
  if (SegmentedScanGraphNode(node.kind)) {
    return AdmitSegmentedScanNode(node, compile_data);
  }
  if (SegmentedReduceGraphNode(node.kind)) {
    return AdmitSegmentedReduceNode(node, compile_data);
  }
  if (SortGraphNode(node.kind)) {
    return AdmitSortNode(node, compile_data);
  }
  if (CompactGraphNode(node.kind)) {
    return AdmitCompactNode(node, compile_data);
  }
  if (GatherGraphNode(node.kind)) {
    return AdmitGatherNode(node, compile_data);
  }
  if (HistogramGraphNode(node.kind)) {
    return AdmitHistogramNode(node, compile_data);
  }
  if (PartitionGraphNode(node.kind)) {
    return AdmitPartitionNode(node, compile_data);
  }
  if (ReduceGraphNode(node.kind)) {
    return AdmitReduceNode(node, compile_data);
  }
  if (ScatterGraphNode(node.kind)) {
    return AdmitScatterNode(node, compile_data);
  }
  if (ScatterReduceGraphNode(node.kind)) {
    return AdmitScatterReduceNode(node, compile_data);
  }
  if (StencilGraphNode(node.kind)) {
    return AdmitStencilNode(node, compile_data);
  }
  if (TransformGraphNode(node.kind)) {
    return AdmitTransformNode(node, compile_data);
  }
  if (MatrixGraphNode(node.kind)) {
    return AdmitMatrixNode(node, compile_data);
  }
  if (FactorGraphNode(node.kind)) {
    return AdmitFactorNode(node, compile_data);
  }
  if (SolveGraphNode(node.kind)) {
    return AdmitSolveNode(node, compile_data);
  }
  if (SpectrumGraphNode(node.kind)) {
    return AdmitSpectrumNode(node, compile_data);
  }
  return "accel_kernel_graph_invalid";
}

} // namespace rund::node::accel::detail
