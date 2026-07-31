#pragma once

#include <accel/graph/node.hpp>

#include <node/accel/context.hpp>

#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/gather/model.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/histogram/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/sort/model.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/transform/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool
NodeSignatureOk(const rund::AccelGraphNode &node,
                const rund::kernel::GraphSignature &expected) noexcept;

[[nodiscard]] bool ScanBindingsOk(const rund::AccelGraphNode &node,
                                  const rund::kernel::ScanPlan &plan) noexcept;

[[nodiscard]] bool
SegmentedScanBindingsOk(const rund::AccelGraphNode &node,
                        const rund::kernel::SegmentedScanPlan &plan) noexcept;

[[nodiscard]] bool SegmentedReduceBindingsOk(
    const rund::AccelGraphNode &node,
    const rund::kernel::SegmentedReducePlan &plan) noexcept;

[[nodiscard]] bool SortBindingsOk(const rund::AccelGraphNode &node,
                                  const rund::kernel::SortPlan &plan) noexcept;

[[nodiscard]] bool
CompactBindingsOk(const rund::AccelGraphNode &node,
                  const rund::kernel::CompactPlan &plan) noexcept;

[[nodiscard]] bool
GatherBindingsOk(const rund::AccelGraphNode &node,
                 const rund::kernel::GatherPlan &plan) noexcept;

[[nodiscard]] bool
HistogramBindingsOk(const rund::AccelGraphNode &node,
                    const rund::kernel::HistogramPlan &plan) noexcept;

[[nodiscard]] bool
PartitionBindingsOk(const rund::AccelGraphNode &node,
                    const rund::kernel::PartitionPlan &plan) noexcept;

[[nodiscard]] bool
ReduceBindingsOk(const rund::AccelGraphNode &node,
                 const rund::kernel::ReducePlan &plan) noexcept;

[[nodiscard]] bool
ScatterBindingsOk(const rund::AccelGraphNode &node,
                  const rund::kernel::ScatterPlan &plan) noexcept;
[[nodiscard]] bool ScatterReduceBindingsOk(
    const rund::AccelGraphNode &node,
    const rund::kernel::ScatterReducePlan &plan) noexcept;

[[nodiscard]] bool
StencilBindingsOk(const rund::AccelGraphNode &node,
                  const rund::kernel::StencilPlan &plan) noexcept;

[[nodiscard]] bool
TransformBindingsOk(const rund::AccelGraphNode &node,
                    const rund::kernel::TransformPlan &plan) noexcept;

[[nodiscard]] bool
MatrixBindingsOk(const rund::AccelGraphNode &node,
                 const rund::kernel::MatrixPlan &plan) noexcept;

[[nodiscard]] bool
FactorBindingsOk(const rund::AccelGraphNode &node,
                 const rund::kernel::FactorPlan &plan) noexcept;

[[nodiscard]] bool
SolveBindingsOk(const rund::AccelGraphNode &node,
                const rund::kernel::SolvePlan &plan) noexcept;

[[nodiscard]] bool
SpectrumBindingsOk(const rund::AccelGraphNode &node,
                   const rund::kernel::SpectrumPlan &plan) noexcept;

} // namespace rund::node::accel::detail
