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

#include "ref.hpp"

namespace node_accel_contract::scatter::match {

template <typename T> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer values{};
  rund::AccelBuffer index{};
  rund::AccelBuffer output{};
  rund::AccelKernel kernel{};
};

template <typename T>
[[nodiscard]] Resources<T> BuildResources(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ScatterElement element, const std::array<T, 4u> &values,
    const std::array<rund::kernel::u32, 4u> &indices,
    const std::array<T, 6u> &initial_output) {
  namespace fix = node_accel_contract::primitive;

  Resources<T> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.values = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), values.size()));
  out.index = rund::node::accel::CreateAccelBuffer(
      out.context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                                   sizeof(rund::kernel::u32), indices.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context, fix::BufferDesc(rund::BufferUsage::ReadWrite, sizeof(T),
                                   initial_output.size()));
  if (!out.values.check.ok || !out.index.check.ok || !out.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.values, values.data(), values.size() * sizeof(T))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.index, indices.data(),
           indices.size() * sizeof(rund::kernel::u32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.output,
                                            initial_output.data(),
                                            initial_output.size() * sizeof(T))
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
      .element = element,
      .element_count = values.size(),
      .output_count = initial_output.size(),
  };
  const rund::kernel::ScatterPlan plan = rund::kernel::PlanScatter(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelScatter(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
                   });
  return out;
}

} // namespace node_accel_contract::scatter::match
