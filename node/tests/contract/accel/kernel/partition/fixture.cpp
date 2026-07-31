#include <accel/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/primitive/partition.hpp>
#include <kernel/program/compute/partition/plan.hpp>

#include "local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>

namespace node_accel_contract::partition {

void Bind(Fixture &fixture) noexcept {
  fixture.refs = {
      rund::AccelGraphBufferRef{
          .buffer = &fixture.flags,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &fixture.values,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  fixture.nodes = {
      rund::AccelPartition(fixture.refs.data(), fixture.refs.size(),
                           fixture.desc),
  };
  fixture.graph = rund::AccelGraph{
      .nodes = fixture.nodes.data(),
      .node_count = fixture.nodes.size(),
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format =
          test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32),
  };
}

Fixture Make(const rund::AccelDevice &pick) {
  namespace p = node_accel_contract::primitive;
  Fixture fixture{};
  fixture.context = rund::node::accel::OpenAccel(pick);
  fixture.flags = rund::node::accel::CreateAccelBuffer(
      fixture.context, p::BufferDesc(rund::BufferUsage::ReadOnly,
                                     sizeof(rund::kernel::u32), 8u));
  fixture.values = rund::node::accel::CreateAccelBuffer(
      fixture.context, p::BufferDesc(rund::BufferUsage::ReadOnly,
                                     sizeof(rund::kernel::u32), 8u));
  fixture.output = rund::node::accel::CreateAccelBuffer(
      fixture.context, p::BufferDesc(rund::BufferUsage::WriteOnly,
                                     sizeof(rund::kernel::u32), 8u));
  fixture.desc = rund::kernel::PartitionDesc{
      .element_count = 8u,
      .flag_bytes = 4u,
      .value_bytes = 4u,
  };
  fixture.plan = rund::kernel::PlanPartition(fixture.desc);
  fixture.hash = rund::kernel::HashPartition(fixture.desc);
  Bind(fixture);
  return fixture;
}

bool CompileReason(const rund::AccelContext &context,
                   const rund::AccelGraph &graph,
                   const std::string_view reason) {
  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(context, graph);
  return !kernel.check.ok && std::string_view{kernel.check.reason} == reason;
}

} // namespace node_accel_contract::partition
