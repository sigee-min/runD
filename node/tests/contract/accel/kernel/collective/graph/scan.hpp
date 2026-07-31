#pragma once

#include <accel/graph/factory/scan/basic.hpp>

#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "../scan/run.hpp"
#include "domain.hpp"

#include "src/accel/context/internal.hpp"

#include <iostream>

namespace node_accel_contract::collective::graph_case {

[[nodiscard]] inline rund::AccelGraphNode
ScanNode(const State &state, const rund::kernel::ScanDesc &desc) {
  return rund::AccelScan(state.refs.data(), state.refs.size(), desc);
}

[[nodiscard]] inline bool ScanGraphProducesPlan(State &state) {
  const rund::kernel::ScanDesc desc = ScanDesc(state.input.count);
  state.nodes[0] = ScanNode(state, desc);
  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(state.context, state.graph);
  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(state.context, kernel);
  if (!kernel.check.ok || !execution.admission.check.ok) {
    std::cerr << "scan graph rejection compile=" << kernel.check.reason
              << " admission=" << execution.admission.check.reason << '\n';
  }
  if (!kernel.check.ok || !execution.admission.check.ok ||
      execution.steps.size() != 1u ||
      execution.steps.front().kind() != rund::kernel::NodeKind::Scan) {
    return false;
  }
  const auto &active =
      execution.steps.front()
          .operation.get<rund::node::accel::detail::operation::Scan>();
  return active.plan.ok && active.plan.pass_count != 0u &&
         active.plan.temp_bytes >=
             state.input.count * sizeof(rund::kernel::u32);
}

} // namespace node_accel_contract::collective::graph_case
