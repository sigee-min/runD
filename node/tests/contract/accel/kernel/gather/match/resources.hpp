#pragma once

#include <accel/graph/factory/primitive/gather.hpp>
#include <kernel/program/compute/gather/plan.hpp>

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

namespace node_accel_contract::gather::match {

template <typename T> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer source{};
  rund::AccelBuffer index{};
  rund::AccelBuffer output{};
  rund::AccelKernel kernel{};
};

template <typename T>
[[nodiscard]] Resources<T> BuildResources(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::GatherElement element, const std::array<T, 6u> &values,
    const std::array<rund::kernel::u32, 4u> &indices) {
  namespace fix = node_accel_contract::primitive;

  Resources<T> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.source = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), values.size()));
  out.index = rund::node::accel::CreateAccelBuffer(
      out.context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                                   sizeof(rund::kernel::u32), indices.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(T), indices.size()));
  if (!out.source.check.ok || !out.index.check.ok || !out.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.source, values.data(), values.size() * sizeof(T))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.index, indices.data(),
           indices.size() * sizeof(rund::kernel::u32))
           .ok) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.source,
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
  const rund::kernel::GatherDesc desc{
      .element = element,
      .element_count = indices.size(),
      .source_count = values.size(),
  };
  const rund::kernel::GatherPlan plan = rund::kernel::PlanGather(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelGather(refs.data(), refs.size(), desc)};
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

} // namespace node_accel_contract::gather::match
