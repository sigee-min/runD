#pragma once

#include <kernel/program/compute/graph/schema.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool ScanGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool SegmentedScanGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool SegmentedReduceGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool SortGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool CompactGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool GatherGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool HistogramGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool PartitionGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool ReduceGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool ScatterGraphNode(
    rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool ScatterReduceGraphNode(
    rund::kernel::NodeKind kind) noexcept;

[[nodiscard]] bool StencilGraphNode(
    rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool TransformGraphNode(
    rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool MatrixGraphNode(
    rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool FactorGraphNode(
    rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool SolveGraphNode(
    rund::kernel::NodeKind kind) noexcept;
[[nodiscard]] bool SpectrumGraphNode(
    rund::kernel::NodeKind kind) noexcept;

}  // namespace rund::node::accel::detail
