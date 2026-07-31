#pragma once

#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

namespace node_accel_contract::gather {

bool RejectsOutOfRangeIndex(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  namespace rej = node_accel_contract::gather::reject;

  if (!pick.check.ok) {
    return false;
  }
  const rej::Work work{};
  if (!rej::ReferenceRejectsOutOfRangeIndex(work)) {
    return false;
  }
  const rej::Resources resources = rej::BuildResources(pick, work);
  const auto bindings = rej::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = work.indices.size(),
                                            .fresh_evidence = true,
                                        });
  if (!resources.plan.ok || !resources.kernel.check.ok ||
      !fix::EvidenceReason(evidence, "compute_gather_index_out_of_range") ||
      evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return false;
  }
  std::array<rund::kernel::u32, 2u> observed{};
  const rund::AccelCheck downloaded =
      rund::node::accel::DownloadAccelBuffer(
          resources.context, resources.output, observed.data(),
          observed.size() * sizeof(rund::kernel::u32));
  return downloaded.ok && observed == work.output_sentinel;
}

bool RejectsBoundedCountOverflowWithoutMutation(
    const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  namespace rej = node_accel_contract::gather::reject;
  if (!pick.check.ok) {
    return false;
  }
  const rej::Work work{};
  const rej::Resources resources = rej::BuildResources(pick, work, true);
  const auto bindings = rej::BoundedBindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = work.indices.size(),
                                            .fresh_evidence = true,
                                        });
  if (!resources.plan.ok || !resources.kernel.check.ok ||
      !fix::EvidenceReason(evidence, "compute_bounded_count_invalid")) {
    return false;
  }
  std::array<rund::kernel::u32, 2u> observed{};
  const rund::AccelCheck downloaded =
      rund::node::accel::DownloadAccelBuffer(
          resources.context, resources.output, observed.data(),
          observed.size() * sizeof(rund::kernel::u32));
  return downloaded.ok && observed == work.output_sentinel;
}

} // namespace node_accel_contract::gather
