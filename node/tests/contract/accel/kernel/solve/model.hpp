#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/factor/node.hpp>
#include <accel/graph/factory/primitive/solve/node.hpp>
#include <kernel/program/compute/solve/reference.hpp>

#include "../primitive/local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <cmath>
#include <iostream>

namespace node_accel_contract::solve_contract {

constexpr rund::kernel::i32 kOne = 0x40000000;
constexpr rund::kernel::i32 kHalf = 0x20000000;
constexpr rund::kernel::i32 kQuarter = 0x10000000;
constexpr rund::kernel::i64 kWideOne = 0x4000000000000000ll;
constexpr rund::kernel::i64 kWideHalf = 0x2000000000000000ll;
constexpr rund::kernel::i64 kWideQuarter = 0x1000000000000000ll;

[[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
FixedFormat(const rund::kernel::ComputeScalar scalar) noexcept {
  return rund::kernel::PrimitiveFixedFormat(
      test::FixedFormatForLane(scalar),
      rund::kernel::ComputeApproximation::Deterministic);
}

[[nodiscard]] inline bool SolveFail(const char *const reason) {
  std::cerr << "solve backend match failed: " << reason << '\n';
  return false;
}

template <typename Value>
[[nodiscard]] bool SolveReuseFail(const char *const reason,
                                  const rund::kernel::FactorOp factor) {
  std::cerr << "solve factor-reuse failed: " << reason
            << " factor=" << static_cast<int>(factor)
            << " bytes=" << sizeof(Value) << '\n';
  return false;
}

template <typename Value, std::size_t Count>
[[nodiscard]] bool OutputMatches(const rund::AccelContext &context,
                                 const rund::AccelBuffer &output,
                                 const std::array<Value, Count> &expected) {
  std::array<Value, Count> actual{};
  if (!rund::node::accel::DownloadAccelBuffer(context, output, actual.data(),
                                              actual.size() * sizeof(Value))
           .ok) {
    return false;
  }
  for (std::size_t i = 0u; i < Count; ++i) {
    const long double scale = sizeof(Value) == sizeof(rund::kernel::i64)
                                  ? 9223372036854775808.0L
                                  : 2147483648.0L;
    const long double diff = std::fabs((static_cast<long double>(actual[i]) -
                                        static_cast<long double>(expected[i])) /
                                       scale);
    if (diff > 1.0e-6L) {
      std::cerr << "solve output mismatch index=" << i
                << " actual=" << actual[i] << " expected=" << expected[i]
                << " normalized_diff=" << static_cast<double>(diff) << '\n';
      return false;
    }
  }
  return true;
}

template <typename Value, std::size_t Count>
[[nodiscard]] bool
ExactOutputMatches(const rund::AccelContext &context,
                   const rund::AccelBuffer &output,
                   const std::array<Value, Count> &expected) {
  std::array<Value, Count> actual{};
  if (!rund::node::accel::DownloadAccelBuffer(context, output, actual.data(),
                                              actual.size() * sizeof(Value))
           .ok) {
    return false;
  }
  for (std::size_t index = 0u; index < Count; ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << "solve exact mismatch index=" << index
                << " actual=" << actual[index]
                << " expected=" << expected[index] << '\n';
      return false;
    }
  }
  return true;
}

template <typename Value>
[[nodiscard]] std::array<Value, 2u> ExpectedIdentitySolve2() {
  return std::array<Value, 2u>{
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne
                                                                    : kOne),
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf
                                                                    : kHalf),
  };
}

template <typename Value>
[[nodiscard]] std::array<Value, 3u> ExpectedIdentitySolve3() {
  return std::array<Value, 3u>{
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne
                                                                    : kOne),
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf
                                                                    : kHalf),
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne
                                                                    : kOne),
  };
}

struct SolveBuffers {
  rund::AccelBuffer matrix{};
  rund::AccelBuffer factor{};
  rund::AccelBuffer aux{};
  rund::AccelBuffer rhs{};
  rund::AccelBuffer output{};
  rund::AccelBuffer status{};
};

template <typename Value>
[[nodiscard]] bool CreateSolveBuffers(const rund::AccelContext &context,
                                      SolveBuffers &b,
                                      const std::uint64_t factor_count = 4u,
                                      const std::uint64_t matrix_count = 4u,
                                      const std::uint64_t rhs_count = 2u,
                                      const std::uint64_t aux_count = 2u,
                                      const std::uint64_t status_count = 1u) {
  namespace fix = node_accel_contract::primitive;
  b.matrix = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value),
                               matrix_count));
  b.factor = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadWrite, sizeof(Value),
                               factor_count));
  b.aux = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadWrite,
                               sizeof(rund::kernel::u32), aux_count));
  b.rhs = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value), rhs_count));
  b.output = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value), rhs_count));
  b.status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), status_count));
  return b.matrix.check.ok && b.factor.check.ok && b.aux.check.ok &&
         b.rhs.check.ok && b.output.check.ok && b.status.check.ok;
}

} // namespace node_accel_contract::solve_contract
