#pragma once

#include <accel/graph/factory/map.hpp>
#include <accel/graph/factory/scan/segmented.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>

#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/visibility.hpp>

#include <node/accel/context.hpp>

#include "test/compute/fixed.hpp"

#include "buffer.hpp"

namespace node_accel_contract::cpu_context::segmented {

[[nodiscard]] inline Resources Compile(Resources out,
                                       const rund::compute_dsl::ComputeOp &op) {
  const rund::kernel::SegmentedScanDesc desc{
      .op = rund::kernel::SegmentedScanOp::ExclusiveSum,
      .element = rund::kernel::SegmentedScanElement::U32,
      .element_count = kCount,
      .block_size = 4u,
};
  const rund::kernel::SegmentedScanPlan plan =
      rund::kernel::PlanSegmentedScan(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphBufferRef, 3u> scan_refs{
      rund::AccelGraphBufferRef{.buffer = &out.read,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.heads,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.mid,
                                .role = rund::kernel::BufferRole::Write,
                                .visibility =
                                    rund::GraphBufferVisibility::Internal}};
  const std::array<rund::AccelGraphBufferRef, 2u> map_refs{
      rund::AccelGraphBufferRef{.buffer = &out.mid,
                                .role = rund::kernel::BufferRole::Read,
                                .binding_name = "scan",
                                .visibility =
                                    rund::GraphBufferVisibility::Internal},
      rund::AccelGraphBufferRef{.buffer = &out.write,
                                .role = rund::kernel::BufferRole::Write,
                                .binding_name = "output"}};
  const std::array<rund::AccelGraphNode, 2u> nodes{
      rund::AccelSegmentedScan(scan_refs.data(), scan_refs.size(), desc),
      rund::AccelMap(op.ir(), map_refs.data(), map_refs.size(),
                     desc.element_count)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context,
      rund::AccelGraph{.nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(
                           rund::kernel::ComputeScalar::Lane32),
});
  return out;
}

} // namespace node_accel_contract::cpu_context::segmented
