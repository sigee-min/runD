#pragma once

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/stencil.hpp>
#include <kernel/program/compute/stencil/plan.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "ref.hpp"

namespace node_accel_contract::stencil::match {

template <typename T, std::size_t Count> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer input{};
  rund::AccelBuffer output{};
  rund::AccelKernel kernel{};
};

template <typename T, std::size_t Count>
[[nodiscard]] Resources<T, Count> BuildResources(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element, const rund::kernel::u64 radius,
    const std::array<T, Count> &input) {
  namespace fix = node_accel_contract::primitive;

  Resources<T, Count> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }

  out.input = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), input.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(T), input.size()));
  if (!out.input.check.ok || !out.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.input, input.data(), input.size() * sizeof(T))
           .ok) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelRead(out.input),
      rund::AccelWrite(out.output),
  };
  const rund::kernel::StencilDesc desc{
      .op = op,
      .element = element,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = input.size(),
      .radius = radius,
  };
  const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  if (!plan.ok) {
    return out;
  }

  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelStencil(refs.data(), refs.size(), desc),
  };
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = domain,
                   });
  return out;
}

} // namespace node_accel_contract::stencil::match
