#pragma once

#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "resources.hpp"

#include <memory>

namespace node_accel_contract::cpu_context {

[[nodiscard]] inline bool RejectsForgedMapKernel(MapRunResources &resources) {
  rund::AccelKernel forged_kernel = resources.kernel;
  forged_kernel.owner = std::make_shared<int>(7);
  const auto bindings = MapRunBindings(resources);
  const rund::AccelEvidence forged = rund::node::accel::RunAccelKernel(
      resources.context, forged_kernel,
      rund::AccelRun{
          .bindings = bindings.data(),
          .binding_count = bindings.size(),
          .tile_count = MapRunResources::kCount,
          .fresh_evidence = true,
      });
  return EvidenceReason(forged, "accel_kernel_run_invalid");
}

[[nodiscard]] inline bool RejectsMapBindingOrder(MapRunResources &resources) {
  const std::array<rund::AccelRunBinding, 2u> wrong_order{
      rund::AccelRunBinding{
          .buffer = &resources.write,
          .role = rund::kernel::BufferRole::Write,
      },
      rund::AccelRunBinding{
          .buffer = &resources.read,
          .role = rund::kernel::BufferRole::Read,
      },
  };
  const rund::AccelEvidence wrong_binding = rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{
          .bindings = wrong_order.data(),
          .binding_count = wrong_order.size(),
          .tile_count = MapRunResources::kCount,
          .fresh_evidence = true,
      });
  return EvidenceReason(wrong_binding, "accel_kernel_run_invalid");
}

} // namespace node_accel_contract::cpu_context
