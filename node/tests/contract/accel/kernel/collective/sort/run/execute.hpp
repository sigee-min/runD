#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

namespace node_accel_contract::collective {

template <typename Key, std::size_t Count>
[[nodiscard]] SortRunHashes<Key>
SortHashesMatchCpuReference(const rund::AccelDevice &pick,
                            const rund::kernel::ComputeScalar scalar,
                            const std::array<Key, Count> &input_keys,
                            const rund::kernel::u32 key_bits = 0u,
                            const rund::kernel::ComputeDomain domain =
                                rund::kernel::ComputeDomain::Fixed) {
  if (!pick.check.ok) {
    return {};
  }
  const bool signed_order = domain == rund::kernel::ComputeDomain::I32 ||
                            domain == rund::kernel::ComputeDomain::I64 ||
                            domain == rund::kernel::ComputeDomain::Fixed;
  const sort_run::Reference<Key, Count> ref =
      sort_run::BuildReference<Key>(input_keys, key_bits, signed_order);
  if (!ref.ok) {
    return {};
  }
  sort_run::Resources<Key, Count> resources =
      sort_run::BuildResources(pick, scalar, domain, input_keys, ref, key_bits);
  if (!resources.kernel.check.ok) {
    return {};
  }
  auto bindings = sort_run::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input_keys.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return {};
  }

  std::array<Key, Count> downloaded_keys{};
  std::array<rund::kernel::u32, Count> downloaded_values{};
  const rund::AccelCheck key_download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write_keys, downloaded_keys.data(),
      downloaded_keys.size() * sizeof(Key));
  const rund::AccelCheck value_download =
      rund::node::accel::DownloadAccelBuffer(
          resources.context, resources.write_values, downloaded_values.data(),
          downloaded_values.size() * sizeof(rund::kernel::u32));
  const std::uint64_t key_hash =
      HashValues(downloaded_keys.data(), downloaded_keys.size());
  const std::uint64_t value_hash =
      HashValues(downloaded_values.data(), downloaded_values.size());
  if (!key_download.ok || !value_download.ok || downloaded_keys != ref.keys ||
      downloaded_values != ref.values ||
      !StableEqualKeyOrderOk(downloaded_keys, downloaded_values, key_bits,
                             signed_order)) {
    return {};
  }
  return SortRunHashes<Key>{
      .key_hash = key_hash,
      .value_hash = value_hash,
      .dispatch_count = evidence.dispatch_count,
      .command_submit_count = evidence.command_submit_count,
      .ok = true,
  };
}

template <typename Key, std::size_t Count>
[[nodiscard]] bool
SortMatchesCpuReference(const rund::AccelDevice &pick,
                        const rund::kernel::ComputeScalar scalar,
                        const std::array<Key, Count> &input_keys,
                        const rund::kernel::u32 key_bits = 0u,
                        const rund::kernel::ComputeDomain domain =
                            rund::kernel::ComputeDomain::Fixed) {
  return SortHashesMatchCpuReference(pick, scalar, input_keys, key_bits, domain)
      .ok;
}

} // namespace node_accel_contract::collective
