#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include <node/accel/context.hpp>

#include "pick.hpp"

#include <array>

namespace node_accel_contract::collective::graph_case {

struct State {
  rund::AccelDevice pick{};
  rund::AccelContext context{};
  rund::AccelBuffer input{};
  rund::AccelBuffer output{};
  rund::compute_dsl::ComputeOp op{};
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  rund::AccelGraph graph{};
};

[[nodiscard]] inline State MakeState(const rund::AccelDevice &pick) {
  State state{};
  state.pick = pick;
  state.context = rund::node::accel::OpenAccel(pick);
  state.input = rund::node::accel::CreateAccelBuffer(
      state.context, BufferDesc(rund::BufferUsage::ReadOnly));
  state.output = rund::node::accel::CreateAccelBuffer(
      state.context, BufferDesc(rund::BufferUsage::WriteOnly));
  state.op = BuildFixedLane32Op();
  state.refs = {rund::AccelGraphBufferRef{
                    .buffer = &state.input,
                    .role = rund::kernel::BufferRole::Read,
                },
                rund::AccelGraphBufferRef{
                    .buffer = &state.output,
                    .role = rund::kernel::BufferRole::Write,
                }};
  state.graph = rund::AccelGraph{
      .nodes = state.nodes.data(),
      .node_count = state.nodes.size(),
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format = state.op.ir().fixed_format,
  };
  return state;
}

[[nodiscard]] inline bool StateOk(const State &state) {
  return state.context.check.ok && state.input.check.ok &&
         state.output.check.ok && state.op.ok();
}

} // namespace node_accel_contract::collective::graph_case
