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

#include "work.hpp"

#include <array>

namespace node_accel_contract::reduce::reject {

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer source{};
  rund::AccelBuffer output{};
  rund::kernel::ReducePlan plan{};
  rund::AccelKernel kernel{};
};

[[nodiscard]] inline Resources BuildResources(const rund::AccelDevice &pick,
                                              const Work &work) {
  namespace fix = node_accel_contract::primitive;

  Resources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.source = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.input.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                                   sizeof(rund::kernel::u32), 1u));
  if (!out.source.check.ok || !out.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.source, work.input.data(),
           work.input.size() * sizeof(rund::kernel::u32))
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
      .op = rund::kernel::ReduceOp::Sum,
      .element = rund::kernel::ReduceElement::U32,
      .element_count = work.input.size(),
      .block_size = 2u,
  };
  out.plan = rund::kernel::PlanReduce(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelReduce(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::U32,
                   });
  return out;
}

} // namespace node_accel_contract::reduce::reject
