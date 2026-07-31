#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/context/buffer.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include <array>

namespace node_accel_contract::kernel_case {

[[nodiscard]] inline rund::AccelGraph
GraphFor(const rund::kernel::ComputeIR &ir, const rund::AccelBuffer &input,
         const rund::AccelBuffer &output,
         std::array<rund::AccelGraphBufferRef, 2u> &refs,
         std::array<rund::AccelGraphNode, 1u> &nodes) {
  refs = {rund::AccelGraphBufferRef{
              .buffer = &input,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelGraphBufferRef{
              .buffer = &output,
              .role = rund::kernel::BufferRole::Write,
          }};
  nodes = {rund::AccelMap(ir, refs.data(), refs.size(), output.count)};
  return rund::AccelGraph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = ir.scalar,
      .domain = ir.domain,
      .fixed_format = ir.fixed_format,
  };
}

} // namespace node_accel_contract::kernel_case
