#pragma once

#include <accel/graph/factory/primitive/scatter.hpp>
#include <kernel/program/compute/scatter/plan.hpp>

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

#include "work.hpp"

#include <array>

namespace node_accel_contract::scatter::reject {

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer values{};
  rund::AccelBuffer index{};
  rund::AccelBuffer output{};
  rund::kernel::ScatterPlan plan{};
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
  out.values = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.values.size()));
  out.index = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.indices.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadWrite, sizeof(rund::kernel::u32),
                      work.output.size()));
  if (!out.values.check.ok || !out.index.check.ok || !out.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.values, work.values.data(),
           work.values.size() * sizeof(rund::kernel::u32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.index, work.indices.data(),
           work.indices.size() * sizeof(rund::kernel::u32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.output, work.output.data(),
           work.output.size() * sizeof(rund::kernel::u32))
           .ok) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.values,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.index,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::ScatterDesc desc{
      .element = rund::kernel::ScatterElement::U32,
      .element_count = work.values.size(),
      .output_count = work.output.size(),
  };
  out.plan = rund::kernel::PlanScatter(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelScatter(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(
                           rund::kernel::ComputeScalar::Lane32),
                   });
  return out;
}

} // namespace node_accel_contract::scatter::reject
