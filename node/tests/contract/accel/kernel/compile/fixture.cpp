#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

namespace node_accel_contract::kernel_case::compile {

Fixture MakeFixture(const rund::AccelDevice &pick) {
  Fixture fixture{};
  fixture.pick = pick;
  fixture.context = rund::node::accel::OpenAccel(pick);
  if (!fixture.context.check.ok) {
    return fixture;
  }
  fixture.input = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::ReadOnly));
  fixture.output = rund::node::accel::CreateAccelBuffer(
      fixture.context, BufferDesc(rund::BufferUsage::WriteOnly));
  fixture.op = BuildFixedLane32Op();
  return fixture;
}

bool Prepare(Fixture &fixture) {
  if (!fixture.context.check.ok || !fixture.input.check.ok ||
      !fixture.output.check.ok || !fixture.op.ok()) {
    return false;
  }
  fixture.graph = GraphFor(fixture.op.ir(), fixture.input, fixture.output,
                           fixture.refs, fixture.nodes);
  fixture.first =
      rund::node::accel::CompileAccelKernel(fixture.context, fixture.graph);
  fixture.second =
      rund::node::accel::CompileAccelKernel(fixture.context, fixture.graph);
  return fixture.first.check.ok && fixture.second.check.ok;
}

} // namespace node_accel_contract::kernel_case::compile
