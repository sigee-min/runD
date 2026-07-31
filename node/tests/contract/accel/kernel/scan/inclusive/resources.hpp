#pragma once

#include <accel/graph/factory/scan/basic.hpp>
#include <kernel/program/compute/scan/plan.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include "ref.hpp"
#include "test/compute/fixed.hpp"

#include <node/accel/context.hpp>

namespace node_accel_contract::scan_inclusive {

template <typename T> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer read{};
  rund::AccelBuffer write{};
  rund::AccelKernel kernel{};
};

template <typename T, std::size_t N>
[[nodiscard]] Resources<T> BuildResources(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ScanElement element, const std::array<T, N> &input) {
  namespace p = node_accel_contract::primitive;

  Resources<T> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.read = rund::node::accel::CreateAccelBuffer(
      out.context,
      p::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), input.size()));
  out.write = rund::node::accel::CreateAccelBuffer(
      out.context,
      p::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(T), input.size()));
  if (!out.read.check.ok || !out.write.check.ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.read, input.data(),
                                            input.size() * sizeof(T))
           .ok) {
    return out;
  }

  std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.read,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.write,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::ScanDesc desc{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = element,
      .element_count = input.size(),
      .block_size = 4u,
  };
  const rund::kernel::ScanPlan plan = rund::kernel::PlanScan(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelScan(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
                   });
  return out;
}

} // namespace node_accel_contract::scan_inclusive
