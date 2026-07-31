#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include "bindings.hpp"

#include <node/accel/context.hpp>

#include <cstdio>

namespace node_accel_contract {

struct InclusiveScanRunCounters {
  rund::kernel::u64 dispatch_count = 0u;
  rund::kernel::u64 command_submit_count = 0u;
};

template <typename T, std::size_t N>
[[nodiscard]] bool InclusiveScanMatchesReference(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ScanElement element, const std::array<T, N> &input,
    InclusiveScanRunCounters *const counters = nullptr) {
  namespace p = node_accel_contract::primitive;
  namespace scan = node_accel_contract::scan_inclusive;
  if (!pick.check.ok) {
    return false;
  }
  const scan::Reference<T, N> ref = scan::BuildReference(element, input);
  scan::Resources<T> resources =
      scan::BuildResources(pick, scalar, element, input);
  if (!ref.ok || !resources.kernel.check.ok) {
    std::fprintf(stderr, "inclusive scan compile: ref=%d kernel=%d reason=%s\n",
                 ref.ok, resources.kernel.check.ok,
                 resources.kernel.check.reason);
    return false;
  }
  const auto bindings = scan::Bindings(resources);
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = input.size(),
                     .fresh_evidence = true,
});
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    std::fprintf(
        stderr,
        "inclusive scan run: ok=%d reason=%s upload=%llu download=%llu\n",
        evidence.ok, evidence.reason,
        static_cast<unsigned long long>(evidence.host_to_device_bytes),
        static_cast<unsigned long long>(evidence.device_to_host_bytes));
    return false;
  }
  if (counters != nullptr) {
    counters->dispatch_count = evidence.dispatch_count;
    counters->command_submit_count = evidence.command_submit_count;
  }
  std::array<T, N> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write, downloaded.data(),
      downloaded.size() * sizeof(T));
  const auto actual_hash = p::HashValues(downloaded.data(), downloaded.size());
  const auto expected_hash =
      p::HashValues(ref.expected.data(), ref.expected.size());
  if (!download.ok || actual_hash != expected_hash) {
    std::fprintf(stderr,
                 "inclusive scan output: download=%d reason=%s actual=%llu "
                 "expected=%llu\n",
                 download.ok, download.reason,
                 static_cast<unsigned long long>(actual_hash),
                 static_cast<unsigned long long>(expected_hash));
    return false;
  }
  return true;
}

} // namespace node_accel_contract
