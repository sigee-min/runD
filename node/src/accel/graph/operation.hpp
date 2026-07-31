#pragma once

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

#include <type_traits>
#include <utility>
#include <variant>

namespace rund::node::accel::detail::operation {

struct Map {
  static constexpr auto kind = rund::kernel::NodeKind::Map;
};

template <rund::kernel::NodeKind Kind, typename Desc, typename Plan>
struct Primitive {
  static constexpr auto kind = Kind;
  Desc desc{};
  Plan plan{};
};

using Scan = Primitive<rund::kernel::NodeKind::Scan, rund::kernel::ScanDesc,
                       rund::kernel::ScanPlan>;
using SegmentedScan =
    Primitive<rund::kernel::NodeKind::SegmentedScan,
              rund::kernel::SegmentedScanDesc, rund::kernel::SegmentedScanPlan>;
using SegmentedReduce = Primitive<rund::kernel::NodeKind::SegmentedReduce,
                                  rund::kernel::SegmentedReduceDesc,
                                  rund::kernel::SegmentedReducePlan>;
using Sort = Primitive<rund::kernel::NodeKind::Sort, rund::kernel::SortDesc,
                       rund::kernel::SortPlan>;
using Compact = Primitive<rund::kernel::NodeKind::Compact,
                          rund::kernel::CompactDesc, rund::kernel::CompactPlan>;
using Gather = Primitive<rund::kernel::NodeKind::Gather,
                         rund::kernel::GatherDesc, rund::kernel::GatherPlan>;
using Histogram =
    Primitive<rund::kernel::NodeKind::Histogram, rund::kernel::HistogramDesc,
              rund::kernel::HistogramPlan>;
using Partition =
    Primitive<rund::kernel::NodeKind::Partition, rund::kernel::PartitionDesc,
              rund::kernel::PartitionPlan>;
using Reduce = Primitive<rund::kernel::NodeKind::Reduce,
                         rund::kernel::ReduceDesc, rund::kernel::ReducePlan>;
using Scatter = Primitive<rund::kernel::NodeKind::Scatter,
                          rund::kernel::ScatterDesc, rund::kernel::ScatterPlan>;
using ScatterReduce =
    Primitive<rund::kernel::NodeKind::ScatterReduce,
              rund::kernel::ScatterReduceDesc, rund::kernel::ScatterReducePlan>;
using Stencil = Primitive<rund::kernel::NodeKind::Stencil,
                          rund::kernel::StencilDesc, rund::kernel::StencilPlan>;
using Transform =
    Primitive<rund::kernel::NodeKind::Transform, rund::kernel::TransformDesc,
              rund::kernel::TransformPlan>;
using Matrix = Primitive<rund::kernel::NodeKind::Matrix,
                         rund::kernel::MatrixDesc, rund::kernel::MatrixPlan>;
using Factor = Primitive<rund::kernel::NodeKind::Factor,
                         rund::kernel::FactorDesc, rund::kernel::FactorPlan>;
using Solve = Primitive<rund::kernel::NodeKind::Solve, rund::kernel::SolveDesc,
                        rund::kernel::SolvePlan>;
using Spectrum =
    Primitive<rund::kernel::NodeKind::Spectrum, rund::kernel::SpectrumDesc,
              rund::kernel::SpectrumPlan>;

} // namespace rund::node::accel::detail::operation

namespace rund::node::accel::detail {

class Operation {
  using Value =
      std::variant<operation::Map, operation::Scan, operation::SegmentedScan,
                   operation::SegmentedReduce, operation::Sort,
                   operation::Compact, operation::Gather, operation::Histogram,
                   operation::Partition, operation::Reduce, operation::Scatter,
                   operation::ScatterReduce, operation::Stencil,
                   operation::Transform, operation::Matrix, operation::Factor,
                   operation::Solve, operation::Spectrum>;

public:
  Operation() = default;

  template <typename T, typename... Args> T &set(Args &&...args) noexcept {
    return value_.template emplace<T>(std::forward<Args>(args)...);
  }

  template <typename T> [[nodiscard]] T &get() noexcept {
    return std::get<T>(value_);
  }

  template <typename T> [[nodiscard]] const T &get() const noexcept {
    return std::get<T>(value_);
  }

  [[nodiscard]] rund::kernel::NodeKind kind() const noexcept {
    return std::visit(
        [](const auto &value) noexcept {
          using T = std::decay_t<decltype(value)>;
          return T::kind;
        },
        value_);
  }

private:
  Value value_{};
};

static_assert(std::is_nothrow_move_constructible_v<Operation>);
static_assert(std::is_nothrow_move_assignable_v<Operation>);

} // namespace rund::node::accel::detail
