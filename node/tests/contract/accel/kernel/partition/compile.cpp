#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include "src/accel/context/internal.hpp"

#include <array>

namespace node_accel_contract::partition {

bool CompileContract() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return true;
  }

  Fixture fixture = Make(pick);
  Bind(fixture);
  if (!fixture.context.check.ok || !fixture.flags.check.ok ||
      !fixture.values.check.ok || !fixture.output.check.ok ||
      !fixture.plan.ok) {
    return false;
  }

  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(fixture.context, fixture.graph);
  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(fixture.context,
                                                         kernel);
  if (!kernel.check.ok || !execution.admission.check.ok ||
      execution.steps.size() != 1u ||
      execution.steps.front().kind() != rund::kernel::NodeKind::Partition) {
    return false;
  }
  const auto &active =
      execution.steps.front()
          .operation.get<rund::node::accel::detail::operation::Partition>();
  if (!active.plan.ok || active.plan.pass_count != fixture.plan.pass_count ||
      active.plan.temp_bytes != fixture.plan.temp_bytes) {
    return false;
  }

  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{
          .buffer = &fixture.flags,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.values,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      fixture.context, kernel,
      rund::AccelRun{
          .bindings = bindings.data(),
          .binding_count = bindings.size(),
          .tile_count = fixture.desc.element_count,
          .fresh_evidence = true,
      });
  if (!evidence.ok) {
    return false;
  }

  rund::AccelGraphNode bad_hash = fixture.nodes[0];
  bad_hash.primitive_hash_lo ^= 1u;
  const rund::AccelGraph bad_hash_graph{
      .nodes = &bad_hash,
      .node_count = 1u,
      .scalar = fixture.graph.scalar,
      .domain = fixture.graph.domain,
      .fixed_format = fixture.graph.fixed_format,
  };
  if (!CompileReason(fixture.context, bad_hash_graph,
                     "accel_kernel_graph_invalid")) {
    return false;
  }

  rund::AccelGraphNode bad_role = fixture.nodes[0];
  std::array<rund::AccelGraphBufferRef, 3u> bad_refs = fixture.refs;
  bad_refs[1].role = rund::kernel::BufferRole::Write;
  bad_role.buffers = bad_refs.data();
  const rund::AccelGraph bad_role_graph{
      .nodes = &bad_role,
      .node_count = 1u,
      .scalar = fixture.graph.scalar,
      .domain = fixture.graph.domain,
      .fixed_format = fixture.graph.fixed_format,
  };
  return CompileReason(fixture.context, bad_role_graph,
                       "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::partition
