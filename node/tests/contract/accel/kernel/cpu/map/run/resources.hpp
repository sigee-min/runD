#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "../../local.hpp"

#include <array>
#include <cstddef>

namespace node_accel_contract::cpu_context {

struct MapRunResources {
  static constexpr std::size_t kCount = 8u;
  rund::AccelContext context{};
  rund::AccelBuffer read{};
  rund::AccelBuffer write{};
  std::array<rund::kernel::i32, kCount> map_input{};
  std::array<rund::kernel::i32, kCount> map_output{};
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  rund::AccelKernel kernel{};
  bool ok = false;
};

[[nodiscard]] inline MapRunResources
MakeMapRunResources(const rund::AccelDevice &pick) {
  MapRunResources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  out.read = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::ReadOnly, MapRunResources::kCount));
  out.write = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::WriteOnly, MapRunResources::kCount));
  const auto body = rund::compute_dsl::bind(MapRunResources::kCount)
                        .fixed<1, 31>()
                        .read<"input">(out.map_input.data())
                        .write<"output">(out.map_output.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-context-map-for-reject")
          .on(body)
          .map([](auto i, auto b) {
            auto in = b.template read<"input">();
            auto out = b.template write<"output">();
            out[i] = in[i] + in[i] + 5;
          });
  if (!out.context.check.ok || !out.read.check.ok || !out.write.check.ok ||
      !op.ok()) {
    return out;
  }
  out.refs = {
      rund::AccelGraphBufferRef{
          .buffer = &out.read,
          .role = rund::kernel::BufferRole::Read,
          .binding_name = "input",
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.write,
          .role = rund::kernel::BufferRole::Write,
          .binding_name = "output",
      },
  };
  out.nodes = {rund::AccelMap(op.ir(), out.refs.data(), out.refs.size(),
                              out.write.count)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = out.nodes.data(),
                       .node_count = out.nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = op.ir().fixed_format,
                   });
  out.ok = out.kernel.check.ok;
  return out;
}

[[nodiscard]] inline std::array<rund::AccelRunBinding, 2u>
MapRunBindings(const MapRunResources &resources) {
  return {
      rund::AccelRunBinding{
          .buffer = &resources.read,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.write,
          .role = rund::kernel::BufferRole::Write,
      },
  };
}

} // namespace node_accel_contract::cpu_context
