#pragma once

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

#include <cstdio>

namespace node_accel_contract::cpu_context::segmented {

[[nodiscard]] inline bool
ContextRunsSegmentedScanThenMap(const rund::AccelDevice &pick,
                                const rund::AccelApi expected_backend,
                                const rund::kernel::u64 expected_submits,
                                const rund::kernel::u64 expected_dispatches) {
  Work work{};
  if (!BuildWork(work)) {
    return false;
  }
  const rund::compute_dsl::ComputeOp op = BuildMapOp(work);
  Resources resources = Compile(BuildBuffers(pick, op, work), op);
  if (!resources.kernel.check.ok) {
    return false;
  }
  auto bindings = BuildBindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = segmented::kCount,
                                            .fresh_evidence = true,
                                        });
  if (!evidence.outcome.ok || evidence.identity.backend != expected_backend ||
      evidence.run.work.command_submit_count != expected_submits ||
      evidence.run.work.dispatch_count != expected_dispatches ||
      evidence.run.work.original_dispatch_count != 3u ||
      evidence.run.work.final_dispatch_count != expected_dispatches ||
      evidence.run.transfer.host_to_device_bytes != 0u ||
      evidence.run.transfer.device_to_host_bytes != 0u) {
    std::fprintf(
        stderr,
        "segmented scan/map evidence: ok=%u reason=%s backend=%u/%u "
        "submits=%llu/%llu dispatches=%llu/%llu original=%llu final=%llu "
        "upload=%llu download=%llu\n",
        static_cast<unsigned>(evidence.outcome.ok), evidence.outcome.reason,
        static_cast<unsigned>(evidence.identity.backend),
        static_cast<unsigned>(expected_backend),
        static_cast<unsigned long long>(evidence.run.work.command_submit_count),
        static_cast<unsigned long long>(expected_submits),
        static_cast<unsigned long long>(evidence.run.work.dispatch_count),
        static_cast<unsigned long long>(expected_dispatches),
        static_cast<unsigned long long>(
            evidence.run.work.original_dispatch_count),
        static_cast<unsigned long long>(evidence.run.work.final_dispatch_count),
        static_cast<unsigned long long>(
            evidence.run.transfer.host_to_device_bytes),
        static_cast<unsigned long long>(
            evidence.run.transfer.device_to_host_bytes));
    return false;
  }

  std::array<rund::kernel::i32, kCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0]));
  return download.ok && HashValues(downloaded.data(), downloaded.size()) ==
                            HashValues(work.expected_output.data(),
                                       work.expected_output.size());
}

} // namespace node_accel_contract::cpu_context::segmented
