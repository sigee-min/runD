#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "graph.hpp"

namespace node_accel_contract {

bool ScanThenMapPreservesInternalRoundtrip(const rund::AccelDevice &pick) {
  namespace p = node_accel_contract::primitive;
  if (!pick.check.ok) {
    return false;
  }

  scan_stream::Work work{};
  if (!scan_stream::BuildWork(work)) {
    return false;
  }
  const rund::compute_dsl::ComputeOp op = scan_stream::BuildMapOp(work);
  if (!op.ok()) {
    return false;
  }

  scan_stream::Resources resources =
      scan_stream::BuildResources(pick, op, work);
  if (!resources.kernel.check.ok) {
    return false;
  }
  auto bindings = scan_stream::BuildBindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = scan_stream::kCount,
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok ||
      evidence.internal_producer_consumer_roundtrip_bytes !=
          scan_stream::kCount * sizeof(rund::kernel::u32) * 2u ||
      evidence.external_producer_consumer_roundtrip_bytes != 0u ||
      evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return false;
  }

  std::array<rund::kernel::i32, scan_stream::kCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0]));
  return download.ok && p::HashValues(downloaded.data(), downloaded.size()) ==
                            p::HashValues(work.expected_output.data(),
                                          work.expected_output.size());
}

} // namespace node_accel_contract
