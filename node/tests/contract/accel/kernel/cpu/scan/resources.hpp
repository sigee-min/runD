#pragma once

#include <accel/graph/factory/map.hpp>
#include <accel/graph/factory/scan/basic.hpp>
#include <kernel/program/compute/scan/plan.hpp>

#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/visibility.hpp>

#include <node/accel/context.hpp>

#include "test/compute/fixed.hpp"

#include "buffer.hpp"

namespace node_accel_contract::cpu_context::scan {

[[nodiscard]] inline Resources Compile(Resources out,
                                       const rund::compute_dsl::ComputeOp &op) {
  const rund::kernel::ScanDesc scan_desc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = kCount,
      .block_size = 4u,
  };
  const rund::kernel::ScanPlan plan = rund::kernel::PlanScan(scan_desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> scan_refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.read,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.mid,
          .role = rund::kernel::BufferRole::Write,
          .visibility = rund::GraphBufferVisibility::Internal,
      },
  };
  const std::array<rund::AccelGraphBufferRef, 2u> map_refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.mid,
          .role = rund::kernel::BufferRole::Read,
          .binding_name = "scan",
          .visibility = rund::GraphBufferVisibility::Internal,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.write,
          .role = rund::kernel::BufferRole::Write,
          .binding_name = "output",
      },
  };
  const std::array<rund::AccelGraphNode, 2u> nodes{
      rund::AccelScan(scan_refs.data(), scan_refs.size(), scan_desc),
      rund::AccelMap(op.ir(), map_refs.data(), map_refs.size(),
                     scan_desc.element_count),
  };
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(
                           rund::kernel::ComputeScalar::Lane32),
                   });
  return out;
}

} // namespace node_accel_contract::cpu_context::scan
