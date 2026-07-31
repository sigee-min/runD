#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/context/buffer.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include "local.hpp"

#include <kernel/program/compute/dsl.hpp>

#include <array>

namespace node_accel_contract::context_forgery {

[[nodiscard]] rund::compute_dsl::ComputeOp BuildFixedLane32Op() {
  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .param<"dt">(7)
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("node-context-forged-pick-owner")
      .on(body)
      .map([](auto i, auto b) {
        auto dt = b.template param<"dt">();
        auto input = b.template read<"input">();
        auto output = b.template write<"output">();
        output[i] = input[i] + dt;
      });
}

[[nodiscard]] rund::AccelGraph
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

} // namespace node_accel_contract::context_forgery
