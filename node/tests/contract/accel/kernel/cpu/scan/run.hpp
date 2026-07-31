#pragma once

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

namespace node_accel_contract::cpu_context {

[[nodiscard]] bool CpuContextRunsScanThenMap(const rund::AccelDevice &pick) {
  scan::Work work{};
  if (!scan::BuildWork(work)) {
    return false;
  }
  const rund::compute_dsl::ComputeOp op = scan::BuildMapOp(work);
  scan::Resources resources =
      scan::Compile(scan::BuildBuffers(pick, op, work), op);
  if (!resources.kernel.check.ok) {
    return false;
  }
  auto bindings = scan::BuildBindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = scan::kCount,
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.backend != rund::AccelApi::Cpu ||
      evidence.command_submit_count != 0u ||
      evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return false;
  }

  std::array<rund::kernel::i32, scan::kCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0]));
  return download.ok && HashValues(downloaded.data(), downloaded.size()) ==
                            HashValues(work.expected_output.data(),
                                       work.expected_output.size());
}

} // namespace node_accel_contract::cpu_context
