#pragma once

#include <accel/graph/factory/map.hpp>

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

[[nodiscard]] bool
CpuContextRejectsForeignBuffer(const rund::AccelDevice &pick) {
  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  const rund::AccelContext other = rund::node::accel::OpenAccel(pick);
  const rund::AccelBuffer read = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::ReadOnly, 8u));
  const rund::AccelBuffer foreign = rund::node::accel::CreateAccelBuffer(
      other, BufferDesc(rund::BufferUsage::WriteOnly, 8u));
  if (!context.check.ok || !other.check.ok || !read.check.ok ||
      !foreign.check.ok) {
    return false;
  }

  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-context-foreign-buffer")
          .on(body)
          .map([](auto i, auto b) {
            auto in = b.template read<"input">();
            auto out = b.template write<"output">();
            out[i] = rund::compute_dsl::quantize<
                1u, 31u, rund::kernel::ComputeRounding::NearestEven,
                rund::kernel::ComputeOverflow::Saturate,
                rund::kernel::ComputeApproximation::Deterministic>(in[i]);
          });
  if (!op.ok()) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &read,
                                .role = rund::kernel::BufferRole::Read,
                                .binding_name = "input"},
      rund::AccelGraphBufferRef{.buffer = &foreign,
                                .role = rund::kernel::BufferRole::Write,
                                .binding_name = "output"},
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMap(op.ir(), refs.data(), refs.size(), read.count)};
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(
                       rund::kernel::ComputeScalar::Lane32),
               });
  return KernelReason(kernel, "accel_kernel_buffer_owner_mismatch");
}

} // namespace node_accel_contract::cpu_context
