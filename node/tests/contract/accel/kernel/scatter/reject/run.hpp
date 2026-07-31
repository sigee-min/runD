#pragma once

#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

namespace node_accel_contract::scatter {

bool RejectsDuplicateIndex(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  namespace rej = node_accel_contract::scatter::reject;

  if (!pick.check.ok) {
    return false;
  }
  const rej::Work work{};
  if (!rej::ReferenceRejectsDuplicateIndex(work)) {
    return false;
  }
  const rej::Resources resources = rej::BuildResources(pick, work);
  const auto bindings = rej::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = work.values.size(),
                                            .fresh_evidence = true,
                                        });
  return resources.plan.ok && resources.kernel.check.ok &&
         fix::EvidenceReason(evidence, "compute_scatter_duplicate_index") &&
         evidence.host_to_device_bytes == 0u &&
         evidence.device_to_host_bytes == 0u;
}

} // namespace node_accel_contract::scatter
