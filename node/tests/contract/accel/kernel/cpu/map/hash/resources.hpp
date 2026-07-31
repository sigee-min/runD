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

#include "op.hpp"

namespace node_accel_contract::cpu_context {

struct MapHashResources {
  rund::AccelContext context{};
  rund::AccelBuffer read{};
  rund::AccelBuffer write{};
  rund::AccelKernel kernel{};
  bool ok = false;
};

[[nodiscard]] inline MapHashResources
MakeMapHashResources(const rund::AccelDevice &pick, const MapHashWork &work,
                     const rund::compute_dsl::ComputeOp &op) {
  MapHashResources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  out.read = rund::node::accel::CreateAccelBuffer(
      out.context, BufferDesc(rund::BufferUsage::ReadOnly, kMapHashCount));
  out.write = rund::node::accel::CreateAccelBuffer(
      out.context, BufferDesc(rund::BufferUsage::WriteOnly, kMapHashCount));
  if (!out.context.check.ok || !out.read.check.ok || !out.write.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.read, work.input.data(),
           work.input.size() * sizeof(work.input[0]))
           .ok) {
    return out;
  }
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &out.read,
                                .role = rund::kernel::BufferRole::Read,
                                .binding_name = "input"},
      rund::AccelGraphBufferRef{.buffer = &out.write,
                                .role = rund::kernel::BufferRole::Write,
                                .binding_name = "output"},
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMap(op.ir(), refs.data(), refs.size(), out.write.count)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = op.ir().fixed_format,
                   });
  out.ok = out.kernel.check.ok;
  return out;
}

} // namespace node_accel_contract::cpu_context
