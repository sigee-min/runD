#include "kind.hpp"

namespace rund::node::accel::detail {

bool ScanGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Scan;
}

bool SegmentedScanGraphNode(
    const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::SegmentedScan;
}

bool SegmentedReduceGraphNode(
    const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::SegmentedReduce;
}

bool SortGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Sort;
}

bool CompactGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Compact;
}

bool GatherGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Gather;
}

bool HistogramGraphNode(
    const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Histogram;
}

bool PartitionGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Partition;
}

bool ReduceGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Reduce;
}

bool ScatterGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Scatter;
}

bool ScatterReduceGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::ScatterReduce;
}

bool StencilGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Stencil;
}

bool TransformGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Transform;
}

bool MatrixGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Matrix;
}

bool FactorGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Factor;
}

bool SolveGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Solve;
}

bool SpectrumGraphNode(const rund::kernel::NodeKind kind) noexcept {
  return kind == rund::kernel::NodeKind::Spectrum;
}

}  // namespace rund::node::accel::detail
