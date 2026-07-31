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
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = segmented::kCount,
                     .fresh_evidence = true,
});
  if (!evidence.ok || evidence.backend != expected_backend ||
      evidence.command_submit_count != expected_submits ||
      evidence.dispatch_count != expected_dispatches ||
      evidence.original_dispatch_count != 3u ||
      evidence.final_dispatch_count != expected_dispatches ||
      evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    std::fprintf(
        stderr,
        "segmented scan/map evidence: ok=%u reason=%s backend=%u/%u "
        "submits=%llu/%llu dispatches=%llu/%llu original=%llu final=%llu "
        "upload=%llu download=%llu\n",
        static_cast<unsigned>(evidence.ok), evidence.reason,
        static_cast<unsigned>(evidence.backend),
        static_cast<unsigned>(expected_backend),
        static_cast<unsigned long long>(evidence.command_submit_count),
        static_cast<unsigned long long>(expected_submits),
        static_cast<unsigned long long>(evidence.dispatch_count),
        static_cast<unsigned long long>(expected_dispatches),
        static_cast<unsigned long long>(evidence.original_dispatch_count),
        static_cast<unsigned long long>(evidence.final_dispatch_count),
        static_cast<unsigned long long>(evidence.host_to_device_bytes),
        static_cast<unsigned long long>(evidence.device_to_host_bytes));
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
