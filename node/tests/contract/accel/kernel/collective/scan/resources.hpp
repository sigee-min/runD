#pragma once

#include <accel/graph/factory/scan/basic.hpp>
#include <kernel/program/compute/scan/plan.hpp>

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
#include "test/compute/fixed.hpp"

namespace node_accel_contract::collective {

template <typename T> struct ScanResources {
  rund::AccelContext context{};
  rund::AccelBuffer read{};
  rund::AccelBuffer write{};
  rund::AccelKernel kernel{};
  bool ok = false;
};

template <typename T>
[[nodiscard]] ScanResources<T> MakeScanResources(
    const rund::AccelDevice &pick, const rund::kernel::ScanElement element,
    const rund::kernel::ComputeScalar scalar, const std::array<T, 8u> &input) {
  ScanResources<T> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.read = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::ReadOnly, sizeof(T), input.size()));
  out.write = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::WriteOnly, sizeof(T), input.size()));
  if (!out.read.check.ok || !out.write.check.ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.read, input.data(),
                                            input.size() * sizeof(T))
           .ok) {
    return out;
  }

  std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &out.read,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.write,
                                .role = rund::kernel::BufferRole::Write},
  };
  const rund::kernel::ScanDesc desc = ScanDesc(input.size(), element);
  const rund::kernel::ScanPlan plan = rund::kernel::PlanScan(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelScan(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
                   });
  out.ok = out.kernel.check.ok && plan.ok;
  return out;
}

} // namespace node_accel_contract::collective
