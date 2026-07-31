#pragma once

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
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/transform/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool DefaultScanDescriptor(
    const rund::kernel::ScanDesc& desc) noexcept;

[[nodiscard]] bool DefaultSegmentedScanDescriptor(
    const rund::kernel::SegmentedScanDesc& desc) noexcept;

[[nodiscard]] bool DefaultSegmentedReduceDescriptor(
    const rund::kernel::SegmentedReduceDesc& desc) noexcept;

[[nodiscard]] bool DefaultGatherDescriptor(
    const rund::kernel::GatherDesc& desc) noexcept;

[[nodiscard]] bool DefaultHistogramDescriptor(
    const rund::kernel::HistogramDesc& desc) noexcept;

[[nodiscard]] bool DefaultPartitionDescriptor(
    const rund::kernel::PartitionDesc& desc) noexcept;

[[nodiscard]] bool DefaultReduceDescriptor(
    const rund::kernel::ReduceDesc& desc) noexcept;

[[nodiscard]] bool DefaultScatterDescriptor(
    const rund::kernel::ScatterDesc& desc) noexcept;
[[nodiscard]] bool DefaultScatterReduceDescriptor(
    const rund::kernel::ScatterReduceDesc &desc) noexcept;

[[nodiscard]] bool DefaultStencilDescriptor(
    const rund::kernel::StencilDesc& desc) noexcept;

[[nodiscard]] bool DefaultTransformDescriptor(
    const rund::kernel::TransformDesc& desc) noexcept;

[[nodiscard]] bool DefaultMatrixDescriptor(
    const rund::kernel::MatrixDesc& desc) noexcept;

[[nodiscard]] bool DefaultFactorDescriptor(
    const rund::kernel::FactorDesc& desc) noexcept;

[[nodiscard]] bool DefaultSolveDescriptor(
    const rund::kernel::SolveDesc& desc) noexcept;

[[nodiscard]] bool DefaultSpectrumDescriptor(
    const rund::kernel::SpectrumDesc& desc) noexcept;

}  // namespace rund::node::accel::detail
