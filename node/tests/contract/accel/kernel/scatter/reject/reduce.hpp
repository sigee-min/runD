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
  std::array<rund::kernel::u32, 259u> values{5u, 7u, 11u, 13u};
  std::array<rund::kernel::u32, 259u> valid_indices{0u, 1u, 0u, 1u};
  std::array<rund::kernel::u32, 259u> invalid_indices{0u, 2u, 0u, 1u};
  std::array<rund::kernel::u32, 2u> output_sentinel{29u, 31u};
  rund::kernel::u32 overflowing_count{260u};

  ReduceWork() {
    invalid_indices[65u] = 2u;
    invalid_indices.back() = 2u;
  }
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

[[nodiscard]] inline ReduceResources
BuildReduceResources(const rund::AccelDevice &pick, const ReduceWork &work,
                     const bool bounded) {
  namespace fix = node_accel_contract::primitive;
  ReduceResources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok)
    return out;

  out.values = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.values.size()));
  out.indices = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.invalid_indices.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadWrite, sizeof(rund::kernel::u32),
                      work.output_sentinel.size()));
  if (bounded) {
    out.count = rund::node::accel::CreateAccelBuffer(
        out.context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                                     sizeof(rund::kernel::u32), 1u));
  }
  const auto &indices = bounded ? work.valid_indices : work.invalid_indices;
  if (!out.values.check.ok || !out.indices.check.ok || !out.output.check.ok ||
      (bounded && !out.count.check.ok) ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.values, work.values.data(), sizeof(work.values))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.indices,
                                            indices.data(), sizeof(indices))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.output,
                                            work.output_sentinel.data(),
                                            sizeof(work.output_sentinel))
           .ok ||
      (bounded && !rund::node::accel::UploadAccelBuffer(
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
      rund::AccelGraphBufferRef{.buffer = bounded ? &out.count : &out.output,
                                .role = bounded
                                            ? rund::kernel::BufferRole::Read
                                            : rund::kernel::BufferRole::Write},
      rund::AccelGraphBufferRef{.buffer = &out.output,
                                .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::ScatterReduceDesc desc{
      .op = rund::kernel::ScatterReduceOp::Sum,
      .domain = rund::kernel::ComputeDomain::U32,
      .element_count = work.values.size(),
      .output_count = work.output_sentinel.size(),
      .count_source = bounded ? rund::kernel::ComputeCountSource::BufferU32
                              : rund::kernel::ComputeCountSource::Descriptor,
  };
  out.plan = rund::kernel::PlanScatterReduce(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{rund::AccelScatterReduce(
      refs.data(), bounded ? refs.size() : refs.size() - 1u, desc)};
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
      rund::AccelRunBinding{.buffer =
                                bounded ? &resources.count : &resources.output,
                            .role = bounded ? rund::kernel::BufferRole::Read
                                            : rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &resources.output,
                            .role = rund::kernel::BufferRole::Write},
  };
}

[[nodiscard]] inline bool
ReferenceScatterReduceFailuresAreAtomic(const ReduceWork &work) {
  const auto exact_plan =
      rund::kernel::PlanScatterReduce(rund::kernel::ScatterReduceDesc{
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

[[nodiscard]] inline bool RunScatterReduceFailure(const rund::AccelDevice &pick,
                                                  const ReduceWork &work,
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
      evidence.run.transfer.host_to_device_bytes != 0u ||
      evidence.run.transfer.device_to_host_bytes != 0u) {
    return false;
  }
  std::array<rund::kernel::u32, 2u> observed{};
  const rund::AccelCheck downloaded = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, observed.data(), sizeof(observed));
  return downloaded.ok && observed == work.output_sentinel;
}

[[nodiscard]] inline bool
ScatterReduceFailuresAreAtomic(const rund::AccelDevice &pick) {
  if (!pick.check.ok)
    return false;
  const ReduceWork work{};
  return ReferenceScatterReduceFailuresAreAtomic(work) &&
         RunScatterReduceFailure(pick, work, false) &&
         RunScatterReduceFailure(pick, work, true);
}

} // namespace node_accel_contract::scatter::reject
