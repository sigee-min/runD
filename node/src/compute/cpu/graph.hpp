#pragma once

#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/gather/model.hpp>
#include <kernel/program/compute/histogram/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/sort/model.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/transform/model.hpp>
#include <rund/compute/abi/model.hpp>
#include <rund/compute/ops.hpp>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct CpuRuntimeValue final {
  Type type{Type::I32};
  std::size_t count{};
};

struct CpuRuntimeMap final {
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  FlowControl control{};
};

struct CpuRuntimeScan final {
  std::uint32_t input{};
  std::uint32_t output{};
  std::uint32_t count{};
  Scan operation{Scan::InclusiveSum};
  FlowControl control{};
};

using CpuRuntimePrimitivePlan =
    std::variant<kernel::SegmentedScanPlan, kernel::SegmentedReducePlan,
                 kernel::SortPlan, kernel::CompactPlan, kernel::GatherPlan,
                 kernel::HistogramPlan, kernel::PartitionPlan,
                 kernel::ReducePlan, kernel::ScatterPlan,
                 kernel::ScatterReducePlan, kernel::StencilPlan,
                 kernel::TransformPlan, kernel::MatrixPlan, kernel::FactorPlan,
                 kernel::SolvePlan, kernel::SpectrumPlan>;

struct CpuRuntimePrimitive final {
  std::vector<std::uint32_t> inputs;
  std::uint32_t output{};
  Primitive kind{Primitive::Reduce};
  CpuRuntimePrimitivePlan plan{};
  FlowControl control{};
};

using CpuRuntimeStep =
    std::variant<CpuRuntimeMap, CpuRuntimeScan, CpuRuntimePrimitive>;

struct CpuRuntimeGraph final {
  std::vector<CpuRuntimeValue> values;
  std::vector<CpuRuntimeStep> steps;
};

} // namespace rund::compute::detail
