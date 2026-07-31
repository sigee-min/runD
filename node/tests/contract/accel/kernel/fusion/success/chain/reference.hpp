#pragma once

#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "fused.hpp"

#include <array>

namespace node_accel_contract::fusion::chain {

[[nodiscard]] bool RunReference(const rund::AccelContext &context,
                                const rund::compute_dsl::ComputeOp &op,
                                const Inputs &inputs,
                                const Resources &resources) {
  std::array<GraphBufferRef, 2u> first_refs{};
  std::array<GraphNode, 1u> first_nodes{};
  const rund::AccelGraph first_graph = GraphFor(
      op.ir(), resources.ref_input, resources.ref_mid, first_refs, first_nodes);
  std::array<GraphBufferRef, 2u> second_refs{};
  std::array<GraphNode, 1u> second_nodes{};
  const rund::AccelGraph second_graph =
      GraphFor(op.ir(), resources.ref_mid, resources.ref_output, second_refs,
               second_nodes);
  const rund::AccelKernel first_kernel =
      rund::node::accel::CompileAccelKernel(context, first_graph);
  const rund::AccelKernel second_kernel =
      rund::node::accel::CompileAccelKernel(context, second_graph);
  if (!first_kernel.check.ok || !second_kernel.check.ok) {
    return false;
  }

  std::array<KernelBinding, 2u> first_bindings{
      KernelBinding{.buffer = &resources.ref_input, .role = Role::Read},
      KernelBinding{.buffer = &resources.ref_mid, .role = Role::Write},
  };
  std::array<KernelBinding, 2u> second_bindings{
      KernelBinding{.buffer = &resources.ref_mid, .role = Role::Read},
      KernelBinding{.buffer = &resources.ref_output, .role = Role::Write},
  };
  return rund::node::accel::RunAccelKernel(
             context, first_kernel,
             rund::AccelRun{.bindings = first_bindings.data(),
                            .binding_count = first_bindings.size(),
                            .tile_count = inputs.host.size(),
                            .fresh_evidence = true,
})
             .ok &&
         rund::node::accel::RunAccelKernel(
             context, second_kernel,
             rund::AccelRun{.bindings = second_bindings.data(),
                            .binding_count = second_bindings.size(),
                            .tile_count = inputs.host.size(),
                            .fresh_evidence = true,
})
             .ok;
}

} // namespace node_accel_contract::fusion::chain
