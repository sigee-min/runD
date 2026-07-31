#pragma once

#include <accel/graph/factory/scan/basic.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "test/compute/fixed.hpp"

#include "../local.hpp"

#include <array>

namespace node_accel_contract::cpu_context {

[[nodiscard]] bool CpuContextRejectsBadScanHash(const rund::AccelDevice &pick) {
  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  const rund::AccelBuffer read = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::ReadOnly, 8u));
  const rund::AccelBuffer write = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::WriteOnly, 8u));
  if (!context.check.ok || !read.check.ok || !write.check.ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &read,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &write,
                                .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::ScanDesc desc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 8u,
      .block_size = 4u,
  };
  rund::AccelGraphNode node = rund::AccelScan(refs.data(), refs.size(), desc);
  node.primitive_hash_lo ^= 1u;
  const std::array<rund::AccelGraphNode, 1u> nodes{node};
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(
                       rund::kernel::ComputeScalar::Lane32),
               });
  return KernelReason(kernel, "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::cpu_context
