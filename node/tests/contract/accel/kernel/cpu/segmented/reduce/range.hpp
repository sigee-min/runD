#pragma once

#include <accel/graph/factory/primitive/segmented/reduce.hpp>
#include <kernel/program/compute/reduce/reference.hpp>
#include <kernel/program/compute/segmented/reduce/reference.hpp>

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "test/compute/fixed.hpp"

#include "../../local.hpp"

#include <array>
#include <cstddef>
#include <cstdio>

namespace node_accel_contract::cpu_context {
namespace segmented {

[[nodiscard]] inline bool ContextRunsSegmentedReduceRangeDefault(
    const rund::AccelDevice &pick, const rund::AccelApi expected_backend,
    const rund::kernel::u64 expected_submits,
    const rund::kernel::u64 expected_dispatches) {
  constexpr std::size_t kCount = 5u;
  std::array<rund::kernel::u32, kCount> input{2u, 1u, 4u, 3u, 5u};
  std::array<rund::kernel::u32, kCount> heads{1u, 1u, 1u, 1u, 1u};
  std::array<rund::kernel::u32, kCount> expected{};
  const rund::kernel::SegmentedReduceResult reference =
      rund::kernel::ReferenceSegmentedReduceSumU32(
          input.data(), heads.data(), expected.data(), input.size());
  if (!reference.ok || reference.segment_count != kCount) {
    return false;
  }

  rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  rund::AccelBuffer values = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(rund::kernel::u32),
                   .count = kCount,
                   .usage = rund::BufferUsage::ReadOnly,
               });
  rund::AccelBuffer segment_heads = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(rund::kernel::u32),
                   .count = kCount,
                   .usage = rund::BufferUsage::ReadOnly,
               });
  rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(rund::kernel::u32),
                   .count = kCount,
                   .usage = rund::BufferUsage::WriteOnly,
               });
  if (!context.check.ok || !values.check.ok || !segment_heads.check.ok ||
      !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, values, input.data(),
                                            input.size() * sizeof(input[0]))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, segment_heads,
                                            heads.data(),
                                            heads.size() * sizeof(heads[0]))
           .ok) {
    return false;
  }

  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{.buffer = &values,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &segment_heads,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write}};
  const rund::AccelGraphNode node = rund::AccelSegmentedReduce(
      refs.data(), refs.size(), rund::kernel::ReduceOp::Sum, kCount);
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = &node,
                   .node_count = 1u,
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(
                       rund::kernel::ComputeScalar::Lane32),
               });
  if (!kernel.check.ok) {
    std::fprintf(stderr,
                 "segmented reduce range compile failed backend=%u reason=%s\n",
                 static_cast<unsigned>(expected_backend), kernel.check.reason);
    return false;
  }

  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &values,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &segment_heads,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output,
                            .role = rund::kernel::BufferRole::Write}};
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = kCount,
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.backend != expected_backend ||
      evidence.command_submit_count != expected_submits ||
      evidence.dispatch_count != expected_dispatches ||
      evidence.original_dispatch_count != 1u ||
      evidence.final_dispatch_count != expected_dispatches) {
    std::fprintf(
        stderr,
        "segmented reduce range evidence backend=%u ok=%u reason=%s "
        "submit=%llu/%llu dispatch=%llu/%llu original=%llu final=%llu\n",
        static_cast<unsigned>(expected_backend), evidence.ok ? 1u : 0u,
        evidence.reason,
        static_cast<unsigned long long>(evidence.command_submit_count),
        static_cast<unsigned long long>(expected_submits),
        static_cast<unsigned long long>(evidence.dispatch_count),
        static_cast<unsigned long long>(expected_dispatches),
        static_cast<unsigned long long>(evidence.original_dispatch_count),
        static_cast<unsigned long long>(evidence.final_dispatch_count));
    return false;
  }

  std::array<rund::kernel::u32, kCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      context, output, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0]));
  return download.ok &&
         HashValues(downloaded.data(), reference.segment_count) ==
             HashValues(expected.data(), reference.segment_count);
}

} // namespace segmented

[[nodiscard]] inline bool
CpuContextRunsSegmentedReduceRangeDefault(const rund::AccelDevice &pick) {
  return segmented::ContextRunsSegmentedReduceRangeDefault(
      pick, rund::AccelApi::Cpu, 0u, 1u);
}

} // namespace node_accel_contract::cpu_context
