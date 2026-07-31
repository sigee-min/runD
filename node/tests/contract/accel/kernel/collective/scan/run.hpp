#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <node/accel/context.hpp>

#include "resources.hpp"

#include <cstdio>

namespace node_accel_contract::collective {

template <typename T>
[[nodiscard]] bool ScanMatchesCpuReference(
    const rund::AccelDevice &pick, const rund::kernel::ScanElement element,
    const rund::kernel::ComputeScalar scalar, const std::array<T, 8u> &input) {
  if (!pick.check.ok) {
    std::fprintf(stderr, "scan pick failed: %s\n", pick.check.reason);
    return false;
  }
  std::array<T, 8u> expected{};
  if (!BuildScanReference(element, input, expected)) {
    std::fprintf(stderr, "scan reference failed: width=%zu\n", sizeof(T));
    return false;
  }
  const ScanResources<T> scan = MakeScanResources(pick, element, scalar, input);
  if (!scan.ok) {
    std::fprintf(stderr,
                 "scan setup failed: api=%u context=%s read=%s write=%s "
                 "kernel=%s width=%zu\n",
                 static_cast<unsigned>(pick.api), scan.context.check.reason,
                 scan.read.check.reason, scan.write.check.reason,
                 scan.kernel.check.reason, sizeof(T));
    return false;
  }

  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{.buffer = &scan.read,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &scan.write,
                            .role = rund::kernel::BufferRole::Write},
  };
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(scan.context, scan.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    std::fprintf(stderr,
                 "scan run failed: api=%u reason=%s upload=%llu download=%llu "
                 "width=%zu\n",
                 static_cast<unsigned>(pick.api), evidence.reason,
                 static_cast<unsigned long long>(evidence.host_to_device_bytes),
                 static_cast<unsigned long long>(evidence.device_to_host_bytes),
                 sizeof(T));
    return false;
  }
  std::array<T, 8u> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      scan.context, scan.write, downloaded.data(),
      downloaded.size() * sizeof(T));
  const std::uint64_t actual_hash =
      HashValues(downloaded.data(), downloaded.size());
  const std::uint64_t expected_hash =
      HashValues(expected.data(), expected.size());
  if (!download.ok || actual_hash != expected_hash) {
    std::fprintf(stderr,
                 "scan output failed: api=%u reason=%s actual=%llu "
                 "expected=%llu width=%zu\n",
                 static_cast<unsigned>(pick.api), download.reason,
                 static_cast<unsigned long long>(actual_hash),
                 static_cast<unsigned long long>(expected_hash), sizeof(T));
    return false;
  }
  return true;
}

} // namespace node_accel_contract::collective
