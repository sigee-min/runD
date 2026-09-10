#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/factory/primitive/scatter/reduce/node.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <kernel/program/compute/scatter/reduce/plan.hpp>
#include <kernel/program/compute/scatter/reduce/reference.hpp>

#include <node/accel/context.hpp>

#include "../../primitive/local.hpp"

#include <array>
#include <bit>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace node_accel_contract::scatter::match {

template <class T, std::size_t N, std::size_t O>
[[nodiscard]] inline bool RunScatterReduceSuccess(
    const rund::AccelDevice &pick, const std::array<T, N> &values,
    const std::array<rund::kernel::u32, N> &indices,
    const std::array<T, O> &expected, const rund::kernel::ScatterReduceOp op,
    const rund::kernel::ComputeDomain domain,
    const rund::kernel::ComputeFixedFormat fixed_format = {}) {
  namespace fix = node_accel_contract::primitive;
  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok)
    return false;
  const rund::AccelBuffer input = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), N));
  const rund::AccelBuffer targets = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::u32), N));
  const rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadWrite, sizeof(T), O));
  if (!input.check.ok || !targets.check.ok || !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, input, values.data(),
                                            sizeof(values))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, targets, indices.data(),
                                            sizeof(indices))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{.buffer = &input,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &targets,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::ScatterReduceDesc desc{
      .op = op,
      .domain = domain,
      .fixed_format = fixed_format,
      .element_count = N,
      .output_count = O,
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelScatterReduce(refs.data(), refs.size(), desc)};
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context,
      rund::AccelGraph{
          .nodes = nodes.data(),
          .node_count = nodes.size(),
          .scalar = sizeof(T) == 8u ? rund::kernel::ComputeScalar::Lane64
                                    : rund::kernel::ComputeScalar::Lane32,
          .domain = domain,
          .fixed_format = fixed_format,
      });
  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &input,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &targets,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output,
                            .role = rund::kernel::BufferRole::Write},
  };
  if (!rund::kernel::PlanScatterReduce(desc).ok || !kernel.check.ok) {
    return false;
  }
  for (unsigned repeat = 0u; repeat != 3u; ++repeat) {
    const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
        context, kernel,
        rund::AccelRun{.bindings = bindings.data(),
                       .binding_count = bindings.size(),
                       .tile_count = N,
                       .fresh_evidence = true});
    std::array<T, O> observed{};
    if (!evidence.outcome.ok ||
        !rund::node::accel::DownloadAccelBuffer(
             context, output, observed.data(), sizeof(observed))
             .ok ||
        observed != expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
ScatterReduceParallelModes(const rund::AccelDevice &pick) {
  using rund::kernel::ComputeDomain;
  using rund::kernel::ScatterReduceOp;
  constexpr std::array<rund::kernel::u32, 4u> indices{0u, 0u, 1u, 1u};
  constexpr std::array<rund::kernel::u32, 4u> sum_values{
      std::numeric_limits<rund::kernel::u32>::max(), 2u, 5u, 7u};
  constexpr std::array<rund::kernel::u32, 2u> sum_expected{1u, 12u};
  constexpr std::array<rund::kernel::i32, 4u> signed_values{-5, 7, -2, 3};
  constexpr std::array<rund::kernel::i32, 2u> signed_min{-5, -2};
  constexpr std::array<rund::kernel::i32, 2u> signed_max{7, 3};
  constexpr std::array<rund::kernel::u32, 4u> unsigned_values{5u, 7u, 2u, 3u};
  constexpr std::array<rund::kernel::u32, 2u> unsigned_min{5u, 2u};
  constexpr std::array<rund::kernel::u32, 2u> unsigned_max{7u, 3u};
  constexpr std::array<rund::kernel::i32, 4u> fixed_values{
      -5 * 65536, 7 * 65536, -2 * 65536, 3 * 65536};
  constexpr std::array<rund::kernel::i32, 2u> fixed_min{-5 * 65536, -2 * 65536};
  constexpr std::array<rund::kernel::i32, 2u> fixed_max{7 * 65536, 3 * 65536};
  constexpr rund::kernel::ComputeFixedFormat fixed{
      .integer_bits = 16u,
      .fraction_bits = 16u,
      .rounding = rund::kernel::ComputeRounding::NearestEven,
      .overflow = rund::kernel::ComputeOverflow::Saturate,
      .approximation = rund::kernel::ComputeApproximation::Exact,
  };
  const std::array<bool, 7u> ok{
      RunScatterReduceSuccess(pick, sum_values, indices, sum_expected,
                              ScatterReduceOp::Sum, ComputeDomain::U32),
      RunScatterReduceSuccess(pick, signed_values, indices, signed_min,
                              ScatterReduceOp::Min, ComputeDomain::I32),
      RunScatterReduceSuccess(pick, signed_values, indices, signed_max,
                              ScatterReduceOp::Max, ComputeDomain::I32),
      RunScatterReduceSuccess(pick, unsigned_values, indices, unsigned_min,
                              ScatterReduceOp::Min, ComputeDomain::U32),
      RunScatterReduceSuccess(pick, unsigned_values, indices, unsigned_max,
                              ScatterReduceOp::Max, ComputeDomain::U32),
      RunScatterReduceSuccess(pick, fixed_values, indices, fixed_min,
                              ScatterReduceOp::Min, ComputeDomain::Fixed,
                              fixed),
      RunScatterReduceSuccess(pick, fixed_values, indices, fixed_max,
                              ScatterReduceOp::Max, ComputeDomain::Fixed,
                              fixed),
  };
  return ok[0] && ok[1] && ok[2] && ok[3] && ok[4] && ok[5] && ok[6];
}

template <typename T>
[[nodiscard]] bool
CheckScatterReduceCohorts(const rund::AccelDevice &pick,
                          const rund::kernel::ComputeDomain domain,
                          const rund::kernel::ComputeFixedFormat fixed = {}) {
  using namespace rund::kernel;
  constexpr std::size_t count = 1027u;
  constexpr std::size_t outputs = 1031u;
  std::array<T, count> values{};
  std::array<u32, count> indices{};
  std::array<T, outputs> expected{};
  std::array<u32, count> scratch{};
  using U = std::make_unsigned_t<T>;
  for (const unsigned pattern : {0u, 1u, 2u}) {
    for (std::size_t i = 0u; i < count; ++i) {
      const U bits = i % 3u == 0u   ? U(std::numeric_limits<T>::max())
                     : i % 3u == 1u ? U{10u}
                                    : U{0u} - U{20u};
      values[i] = std::bit_cast<T>(bits);
      indices[i] = pattern == 0u          ? 17u
                   : pattern == 2u        ? static_cast<u32>(i)
                   : (i / 32u) % 2u == 0u ? 17u
                                          : static_cast<u32>(i % 19u);
    }
    for (const auto op :
         {ScatterReduceOp::Sum, ScatterReduceOp::Min, ScatterReduceOp::Max}) {
      const auto plan =
          PlanScatterReduce(ScatterReduceDesc{.op = op,
                                              .domain = domain,
                                              .fixed_format = fixed,
                                              .element_count = count,
                                              .output_count = outputs});
      ScatterReduceResult reference{};
      if constexpr (std::is_same_v<T, u32>) {
        reference = ReferenceScatterReduceU32(values.data(), indices.data(),
                                              expected.data(), count, plan,
                                              scratch.data(), scratch.size());
      } else if constexpr (std::is_same_v<T, i32>) {
        reference =
            domain == ComputeDomain::Fixed
                ? ReferenceScatterReduceFixedI32(values.data(), indices.data(),
                                                 expected.data(), count, plan,
                                                 scratch.data(), scratch.size())
                : ReferenceScatterReduceI32(values.data(), indices.data(),
                                            expected.data(), count, plan,
                                            scratch.data(), scratch.size());
      } else if constexpr (std::is_same_v<T, u64>) {
        reference = ReferenceScatterReduceU64(values.data(), indices.data(),
                                              expected.data(), count, plan,
                                              scratch.data(), scratch.size());
      } else {
        reference = ReferenceScatterReduceI64(values.data(), indices.data(),
                                              expected.data(), count, plan,
                                              scratch.data(), scratch.size());
      }
      if (!reference.ok ||
          !RunScatterReduceSuccess(pick, values, indices, expected, op, domain,
                                   fixed)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] inline bool ScatterReduceCohorts(const rund::AccelDevice &pick) {
  using namespace rund::kernel;
  const ComputeFixedFormat saturate{.integer_bits = 16u,
                                    .fraction_bits = 16u,
                                    .rounding = ComputeRounding::NearestEven,
                                    .overflow = ComputeOverflow::Saturate,
                                    .approximation =
                                        ComputeApproximation::Exact};
  return CheckScatterReduceCohorts<u32>(pick, ComputeDomain::U32) &&
         CheckScatterReduceCohorts<i32>(pick, ComputeDomain::I32) &&
         CheckScatterReduceCohorts<u64>(pick, ComputeDomain::U64) &&
         CheckScatterReduceCohorts<i64>(pick, ComputeDomain::I64) &&
         CheckScatterReduceCohorts<i32>(pick, ComputeDomain::Fixed, saturate);
}

} // namespace node_accel_contract::scatter::match
