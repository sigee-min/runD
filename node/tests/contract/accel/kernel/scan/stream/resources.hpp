#pragma once

#include <accel/graph/factory/map.hpp>
#include <accel/graph/factory/scan/basic.hpp>
#include <kernel/program/compute/scan/plan.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "work.hpp"

#include <array>
#include <cstdint>

namespace node_accel_contract::scan_stream {

[[nodiscard]] constexpr rund::kernel::ScanDesc ScanDesc() noexcept {
  return rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = kCount,
      .block_size = 4u,
  };
}

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer read{};
  rund::AccelBuffer mid{};
  rund::AccelBuffer write{};
  rund::AccelKernel kernel{};
};

[[nodiscard]] inline Resources
BuildResources(const rund::AccelDevice &pick,
               const rund::compute_dsl::ComputeOp &op, const Work &work) {
  namespace p = node_accel_contract::primitive;
  Resources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.read = rund::node::accel::CreateAccelBuffer(
      out.context, p::BufferDesc(rund::BufferUsage::ReadOnly,
                                 sizeof(rund::kernel::u32), kCount));
  out.mid = rund::node::accel::CreateAccelBuffer(
      out.context, p::BufferDesc(rund::BufferUsage::ReadWrite,
                                 sizeof(rund::kernel::u32), kCount));
  out.write = rund::node::accel::CreateAccelBuffer(
      out.context, p::BufferDesc(rund::BufferUsage::WriteOnly,
                                 sizeof(rund::kernel::i32), kCount));
  if (!out.read.check.ok || !out.mid.check.ok || !out.write.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.read, work.input.data(),
           work.input.size() * sizeof(work.input[0]))
           .ok) {
    return out;
  }

  const rund::kernel::ScanDesc scan_desc = ScanDesc();
  const rund::kernel::ScanPlan scan_plan = rund::kernel::PlanScan(scan_desc);
  if (!scan_plan.ok) {
    return out;
  }
  std::array<rund::AccelGraphBufferRef, 2u> scan_refs{
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
  std::array<rund::AccelGraphBufferRef, 2u> map_refs{
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
      rund::AccelMap(op.ir(), map_refs.data(), map_refs.size(), kCount),
  };
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = op.ir().fixed_format,
                   });
  return out;
}

} // namespace node_accel_contract::scan_stream
