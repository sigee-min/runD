#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include <accel/graph/factory/map.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <array>

namespace node_accel_contract::kernel_case::compile {

bool TwoReadBindingOrderRejects(const Fixture &fixture) {
  const rund::AccelBuffer pos = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::ReadOnly));
  const rund::AccelBuffer vel = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::ReadOnly));
  const rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::WriteOnly));
  const rund::compute_dsl::ComputeOp two_read = BuildTwoReadFixedLane32Op();
  if (!pos.check.ok || !vel.check.ok || !output.check.ok || !two_read.ok()) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &pos,
          .role = rund::kernel::BufferRole::Read,
          .binding_name = "pos",
      },
      rund::AccelGraphBufferRef{
          .buffer = &vel,
          .role = rund::kernel::BufferRole::Read,
          .binding_name = "vel",
      },
      rund::AccelGraphBufferRef{
          .buffer = &output,
          .role = rund::kernel::BufferRole::Write,
          .binding_name = "output",
      },
  };
  std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMap(two_read.ir(), refs.data(), refs.size(), output.count)};
  const rund::AccelGraph graph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = two_read.ir().scalar,
      .domain = two_read.ir().domain,
      .fixed_format = two_read.ir().fixed_format,
  };
  if (!rund::node::accel::CompileAccelKernel(fixture.context, graph).check.ok) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 3u> swapped{
      rund::AccelGraphBufferRef{
          .buffer = &vel,
          .role = rund::kernel::BufferRole::Read,
          .binding_name = "vel",
      },
      rund::AccelGraphBufferRef{
          .buffer = &pos,
          .role = rund::kernel::BufferRole::Read,
          .binding_name = "pos",
      },
      rund::AccelGraphBufferRef{
          .buffer = &output,
          .role = rund::kernel::BufferRole::Write,
          .binding_name = "output",
      },
  };
  nodes[0].buffers = swapped.data();
  if (!CheckReason(
          rund::node::accel::CompileAccelKernel(fixture.context, graph).check,
          "accel_kernel_graph_invalid")) {
    return false;
  }

  const rund::AccelBuffer invalid_owner{};
  swapped = refs;
  swapped[0].buffer = &invalid_owner;
  swapped[2].binding_name = "bad";
  return CheckReason(
      rund::node::accel::CompileAccelKernel(fixture.context, graph).check,
      "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::kernel_case::compile
