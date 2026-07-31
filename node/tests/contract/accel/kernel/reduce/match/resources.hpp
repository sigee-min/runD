#pragma once

#include <accel/graph/factory/primitive/reduce.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

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

#include <span>

namespace node_accel_contract::reduce::match {

template <typename T> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer source{};
  rund::AccelBuffer output{};
  rund::AccelKernel kernel{};
};

template <typename T>
[[nodiscard]] Resources<T> BuildResources(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain, const rund::kernel::ReduceOp op,
    const rund::kernel::ReduceElement element, const std::span<const T> input,
    const rund::kernel::u64 block_size = 4u) {
  namespace fix = node_accel_contract::primitive;
  Resources<T> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.source = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), input.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(T), 1u));
  if (!out.source.check.ok || !out.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.source, input.data(), input.size() * sizeof(T))
           .ok) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.source,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::ReduceDesc desc{
      .op = op,
      .element = element,
      .element_count = input.size(),
      .block_size = block_size,
  };
  const rund::kernel::ReducePlan plan = rund::kernel::PlanReduce(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelReduce(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = domain,
                   });
  return out;
}

} // namespace node_accel_contract::reduce::match
