#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>


#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/gather/model.hpp>
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
#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

struct ScanBinds;
struct SegmentedScanBinds;
struct SegmentedReduceBinds;
struct SortBinds;
struct CompactBinds;
struct GatherBinds;
struct HistogramBinds;
struct PartitionBinds;
struct ReduceBinds;
struct ScatterBinds;
struct ScatterReduceBinds;
struct StencilBinds;
struct TransformBinds;
struct MatrixBinds;
struct FactorBinds;
struct SolveBinds;
struct SpectrumBinds;

[[nodiscard]] rund::AccelCheck ExecuteCpuScan(
    const rund::AccelDevice& pick,
    const rund::kernel::ScanPlan& plan,
    rund::kernel::ComputeDomain domain,
    const ScanBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuSegmentedScan(
    const rund::AccelDevice& pick,
    const rund::kernel::SegmentedScanDesc& desc,
    const rund::kernel::SegmentedScanPlan& plan,
    rund::kernel::ComputeDomain domain,
    const SegmentedScanBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuSegmentedReduce(
    const rund::AccelDevice& pick,
    const rund::kernel::SegmentedReduceDesc& desc,
    const rund::kernel::SegmentedReducePlan& plan,
    rund::kernel::ComputeDomain domain,
    const SegmentedReduceBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuSort(
    const rund::AccelDevice& pick,
    const rund::kernel::SortDesc& desc,
    const rund::kernel::SortPlan& plan,
    rund::kernel::ComputeDomain domain,
    const SortBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuCompact(
    const rund::AccelDevice& pick,
    const rund::kernel::CompactDesc& desc,
    const rund::kernel::CompactPlan& plan,
    const CompactBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuGather(
    const rund::AccelDevice& pick,
    const rund::kernel::GatherDesc& desc,
    const rund::kernel::GatherPlan& plan,
    const GatherBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuHistogram(
    const rund::AccelDevice& pick, const rund::kernel::HistogramDesc& desc,
    const rund::kernel::HistogramPlan& plan,
    const HistogramBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuPartition(
    const rund::AccelDevice& pick,
    const rund::kernel::PartitionDesc& desc,
    const rund::kernel::PartitionPlan& plan,
    const PartitionBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuReduce(
    const rund::AccelDevice& pick,
    const rund::kernel::ReduceDesc& desc,
    const rund::kernel::ReducePlan& plan,
    rund::kernel::ComputeDomain domain,
    const ReduceBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuScatter(
    const rund::AccelDevice& pick,
    const rund::kernel::ScatterDesc& desc,
    const rund::kernel::ScatterPlan& plan,
    const ScatterBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuScatterReduce(
    const rund::AccelDevice &pick, const rund::kernel::ScatterReducePlan &plan,
    const ScatterReduceBinds &bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuStencil(
    const rund::AccelDevice& pick,
    const rund::kernel::StencilDesc& desc,
    const rund::kernel::StencilPlan& plan,
    rund::kernel::ComputeDomain domain,
    const StencilBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuTransform(
    const rund::AccelDevice& pick,
    const rund::kernel::TransformDesc& desc,
    const rund::kernel::TransformPlan& plan,
    const TransformBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuMatrix(
    const rund::AccelDevice& pick,
    const rund::kernel::MatrixDesc& desc,
    const rund::kernel::MatrixPlan& plan,
    const MatrixBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuFactor(
    const rund::AccelDevice& pick,
    const rund::kernel::FactorDesc& desc,
    const rund::kernel::FactorPlan& plan,
    const FactorBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuSolve(
    const rund::AccelDevice& pick,
    const rund::kernel::SolveDesc& desc,
    const rund::kernel::SolvePlan& plan,
    const SolveBinds& bindings);
[[nodiscard]] rund::AccelCheck ExecuteCpuSpectrum(
    const rund::AccelDevice& pick,
    const rund::kernel::SpectrumDesc& desc,
    const rund::kernel::SpectrumPlan& plan,
    const SpectrumBinds& bindings);

}  // namespace rund::node::accel::detail
