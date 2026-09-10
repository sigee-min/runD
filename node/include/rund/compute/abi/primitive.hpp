#pragma once

#include <rund/compute/reason.hpp>
#include <cstdint>

namespace rund::compute::detail {

enum class Primitive : unsigned char {
  SegmentedScan,
  SegmentedReduce,
  Sort,
  Argsort,
  Compact,
  Gather,
  Histogram,
  Partition,
  Reduce,
  Scatter,
  Stencil,
  Transform,
  Matrix,
  Factor,
  Solve,
  Spectrum,
  ScatterReduce,
  Window,
};

[[nodiscard]] constexpr Reason
primitive_execution_reason(const Primitive primitive,
                           const std::uint32_t status) noexcept {
  if (status == 0u) {
    return Reason::Ok;
  }
  switch (primitive) {
  case Primitive::Factor:
    switch (status) {
    case 1u:
      return Reason::FactorSingular;
    case 2u:
      return Reason::FactorNotPositiveDefinite;
    case 3u:
      return Reason::FactorPivotUnderflow;
    case 4u:
      return Reason::FactorScalingInvalid;
    default:
      return Reason::ReasonInvalid;
    }
  case Primitive::Solve:
    switch (status) {
    case 1u:
      return Reason::SolveSingular;
    case 2u:
      return Reason::SolveNotPositiveDefinite;
    case 3u:
      return Reason::SolvePivotUnderflow;
    case 4u:
      return Reason::SolveScalingInvalid;
    default:
      return Reason::ReasonInvalid;
    }
  case Primitive::Spectrum:
    switch (status) {
    case 1u:
      return Reason::SpectrumNonConvergence;
    case 2u:
      return Reason::SpectrumScalingInvalid;
    default:
      return Reason::ReasonInvalid;
    }
  default:
    return Reason::ReasonInvalid;
  }
}

struct PrimitiveOptions final {
  std::uint64_t first{};
  std::uint64_t second{};
  std::uint64_t third{};
  std::uint64_t fourth{};
  std::uint32_t mode{};
  std::uint32_t extra{};
  bool flag{};
};

} // namespace rund::compute::detail
