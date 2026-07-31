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
#include <limits>
#include <string_view>
#include <vector>

namespace node_accel_contract::scatter::reject {

struct ReduceWork final {
  std::array<rund::kernel::u32, 4u> values{5u, 7u, 11u, 13u};
  std::array<rund::kernel::u32, 4u> valid_indices{0u, 1u, 0u, 1u};
  std::array<rund::kernel::u32, 4u> invalid_indices{0u, 2u, 0u, 1u};
  std::array<rund::kernel::u32, 2u> output_sentinel{29u, 31u};
  rund::kernel::u32 overflowing_count{5u};
};

struct ReduceResources final {
  rund::AccelContext context{};
  rund::AccelBuffer values{};
  rund::AccelBuffer indices{};
  rund::AccelBuffer count{};
  rund::AccelBuffer output{};
  rund::kernel::ScatterReducePlan plan{};
  rund::AccelKernel kernel{};
};

[[nodiscard]] inline ReduceResources BuildReduceResources(
    const rund::AccelDevice &pick, const ReduceWork &work,
    const bool bounded) {
  namespace fix = node_accel_contract::primitive;
  ReduceResources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) return out;

  out.values = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly,
                      sizeof(rund::kernel::u32), work.values.size()));
  out.indices = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly,
                      sizeof(rund::kernel::u32), work.invalid_indices.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadWrite,
                      sizeof(rund::kernel::u32), work.output_sentinel.size()));
  if (bounded) {
    out.count = rund::node::accel::CreateAccelBuffer(
        out.context,
        fix::BufferDesc(rund::BufferUsage::ReadOnly,
                        sizeof(rund::kernel::u32), 1u));
  }
  const auto &indices = bounded ? work.valid_indices : work.invalid_indices;
  if (!out.values.check.ok || !out.indices.check.ok || !out.output.check.ok ||
      (bounded && !out.count.check.ok) ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.values, work.values.data(), sizeof(work.values))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.indices, indices.data(), sizeof(indices))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.output, work.output_sentinel.data(),
           sizeof(work.output_sentinel))
           .ok ||
      (bounded &&
       !rund::node::accel::UploadAccelBuffer(
            out.context, out.count, &work.overflowing_count,
            sizeof(work.overflowing_count))
            .ok)) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelGraphBufferRef{.buffer = &out.values,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.indices,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{
          .buffer = bounded ? &out.count : &out.output,
          .role = bounded ? rund::kernel::BufferRole::Read
                          : rund::kernel::BufferRole::Write},
      rund::AccelGraphBufferRef{.buffer = &out.output,
                                .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::ScatterReduceDesc desc{
      .op = rund::kernel::ScatterReduceOp::Sum,
      .domain = rund::kernel::ComputeDomain::U32,
      .element_count = work.values.size(),
      .output_count = work.output_sentinel.size(),
      .count_source = bounded
                          ? rund::kernel::ComputeCountSource::BufferU32
                          : rund::kernel::ComputeCountSource::Descriptor,
  };
  out.plan = rund::kernel::PlanScatterReduce(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelScatterReduce(refs.data(), bounded ? refs.size()
                                                   : refs.size() - 1u,
                               desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::U32,
                   });
  return out;
}

[[nodiscard]] inline std::array<rund::AccelRunBinding, 4u>
ReduceBindings(const ReduceResources &resources, const bool bounded) {
  return {
      rund::AccelRunBinding{.buffer = &resources.values,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &resources.indices,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{
          .buffer = bounded ? &resources.count : &resources.output,
          .role = bounded ? rund::kernel::BufferRole::Read
                          : rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &resources.output,
                            .role = rund::kernel::BufferRole::Write},
  };
}

[[nodiscard]] inline bool ReferenceScatterReduceFailuresAreAtomic(
    const ReduceWork &work) {
  const auto exact_plan = rund::kernel::PlanScatterReduce(
      rund::kernel::ScatterReduceDesc{
          .op = rund::kernel::ScatterReduceOp::Sum,
          .domain = rund::kernel::ComputeDomain::U32,
          .element_count = work.values.size(),
          .output_count = work.output_sentinel.size(),
      });
  std::vector<rund::kernel::u32> scratch(work.values.size());
  auto exact_output = work.output_sentinel;
  const auto invalid_index = rund::kernel::ReferenceScatterReduceU32(
      work.values.data(), work.invalid_indices.data(), exact_output.data(),
      work.values.size(), exact_plan, scratch.data(), scratch.size());
  auto count_output = work.output_sentinel;
  const auto invalid_count = rund::kernel::ReferenceScatterReduceU32(
      work.values.data(), work.valid_indices.data(), count_output.data(),
      work.overflowing_count, exact_plan, scratch.data(), scratch.size());
  return !invalid_index.ok &&
         invalid_index.reason ==
             std::string_view{"compute_scatter_reduce_index_out_of_range"} &&
         exact_output == work.output_sentinel && !invalid_count.ok &&
         invalid_count.reason ==
             std::string_view{"compute_scatter_reduce_count_out_of_range"} &&
         count_output == work.output_sentinel;
}

[[nodiscard]] inline bool RunScatterReduceFailure(
    const rund::AccelDevice &pick, const ReduceWork &work,
    const bool bounded) {
  namespace fix = node_accel_contract::primitive;
  const ReduceResources resources = BuildReduceResources(pick, work, bounded);
  const auto bindings = ReduceBindings(resources, bounded);
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{
          .bindings = bindings.data(),
          .binding_count = bounded ? bindings.size() : bindings.size() - 1u,
          .tile_count = work.values.size(),
          .fresh_evidence = true,
      });
  const std::string_view reason =
      bounded ? "compute_scatter_reduce_count_out_of_range"
              : "compute_scatter_reduce_index_out_of_range";
  if (!resources.plan.ok || !resources.kernel.check.ok ||
      !fix::EvidenceReason(evidence, reason) ||
      evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return false;
  }
  std::array<rund::kernel::u32, 2u> observed{};
  const rund::AccelCheck downloaded = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, observed.data(), sizeof(observed));
  return downloaded.ok && observed == work.output_sentinel;
}

[[nodiscard]] inline bool ScatterReduceFailuresAreAtomic(
    const rund::AccelDevice &pick) {
  if (!pick.check.ok) return false;
  const ReduceWork work{};
  return ReferenceScatterReduceFailuresAreAtomic(work) &&
         RunScatterReduceFailure(pick, work, false) &&
         RunScatterReduceFailure(pick, work, true);
}

template <class T, std::size_t N, std::size_t O>
[[nodiscard]] inline bool RunScatterReduceSuccess(
    const rund::AccelDevice &pick, const std::array<T, N> &values,
    const std::array<rund::kernel::u32, N> &indices,
    const std::array<T, O> &expected,
    const rund::kernel::ScatterReduceOp op,
    const rund::kernel::ComputeDomain domain,
    const rund::kernel::ComputeFixedFormat fixed_format = {}) {
  namespace fix = node_accel_contract::primitive;
  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) return false;
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
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
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
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = N,
                     .fresh_evidence = true});
  std::array<T, O> observed{};
  return rund::kernel::PlanScatterReduce(desc).ok && kernel.check.ok &&
         evidence.ok &&
         rund::node::accel::DownloadAccelBuffer(context, output,
                                                observed.data(),
                                                sizeof(observed))
             .ok &&
         observed == expected;
}

[[nodiscard]] inline bool ScatterReduceParallelModes(
    const rund::AccelDevice &pick) {
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
  constexpr std::array<rund::kernel::i32, 2u> fixed_min{-5 * 65536,
                                                        -2 * 65536};
  constexpr std::array<rund::kernel::i32, 2u> fixed_max{7 * 65536,
                                                        3 * 65536};
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

} // namespace node_accel_contract::scatter::reject
