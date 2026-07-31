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

#include "work.hpp"

#include <array>

namespace node_accel_contract::gather::reject {

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer source{};
  rund::AccelBuffer index{};
  rund::AccelBuffer count{};
  rund::AccelBuffer output{};
  rund::kernel::GatherPlan plan{};
  rund::AccelKernel kernel{};
};

[[nodiscard]] inline Resources BuildResources(const rund::AccelDevice &pick,
                                              const Work &work,
                                              const bool bounded = false) {
  namespace fix = node_accel_contract::primitive;

  Resources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.source = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.values.size()));
  out.index = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      work.indices.size()));
  out.output = rund::node::accel::CreateAccelBuffer(
      out.context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(rund::kernel::u32),
                      work.indices.size()));
  if (bounded) {
    out.count = rund::node::accel::CreateAccelBuffer(
        out.context,
        fix::BufferDesc(rund::BufferUsage::ReadOnly,
                        sizeof(rund::kernel::u32), 1u));
  }
  if (!out.source.check.ok || !out.index.check.ok || !out.output.check.ok ||
      (bounded && !out.count.check.ok) ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.source, work.values.data(),
           work.values.size() * sizeof(rund::kernel::u32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.index, work.indices.data(),
           work.indices.size() * sizeof(rund::kernel::u32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.output, work.output_sentinel.data(),
           work.output_sentinel.size() * sizeof(rund::kernel::u32))
           .ok ||
      (bounded &&
       !rund::node::accel::UploadAccelBuffer(
            out.context, out.count, &work.overflowing_count,
            sizeof(work.overflowing_count))
            .ok)) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.source,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.index,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = bounded ? &out.count : &out.output,
          .role = bounded ? rund::kernel::BufferRole::Read
                          : rund::kernel::BufferRole::Write,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::GatherDesc desc{
      .element = rund::kernel::GatherElement::U32,
      .element_count = work.indices.size(),
      .source_count = work.values.size(),
      .count_source = bounded
                          ? rund::kernel::ComputeCountSource::BufferU32
                          : rund::kernel::ComputeCountSource::Descriptor,
  };
  out.plan = rund::kernel::PlanGather(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelGather(refs.data(), bounded ? refs.size() : refs.size() - 1u,
                        desc)};
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

} // namespace node_accel_contract::gather::reject
