#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/context/buffer.hpp>
#include <accel/graph/value.hpp>

#include "model.hpp"

namespace node_accel_contract::fusion {

[[nodiscard]] inline rund::AccelGraph
GraphFor(const rund::kernel::ComputeIR &ir, const rund::AccelBuffer &input,
         const rund::AccelBuffer &output, std::array<GraphBufferRef, 2u> &refs,
         std::array<GraphNode, 1u> &nodes) {
  refs = {GraphBufferRef{.buffer = &input, .role = Role::Read},
          GraphBufferRef{.buffer = &output, .role = Role::Write}};
  nodes = {rund::AccelMap(ir, refs.data(), refs.size(), output.count)};
  return rund::AccelGraph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = ir.scalar,
      .domain = ir.domain,
      .fixed_format = ir.fixed_format,
  };
}

} // namespace node_accel_contract::fusion
