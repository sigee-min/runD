#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <node/accel/context.hpp>

#include "order.hpp"
#include "ref.hpp"
#include "resources.hpp"

namespace node_accel_contract {

bool SortIdentityU32MatchesCpuReference(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const std::array<rund::kernel::u32, 8u> &input_keys,
    const rund::kernel::u32 key_bits) {
  namespace p = node_accel_contract::primitive;
  if (!pick.check.ok) {
    return false;
  }
  const sort_identity::Expected expected =
      sort_identity::MakeExpected(input_keys);
  sort_identity::Resources resources =
      sort_identity::PrepareResources(pick, scalar, input_keys, key_bits);
  if (!expected.valid || !resources.valid) {
    return false;
  }

  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{
          .buffer = &resources.read_keys,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.write_keys,
          .role = rund::kernel::BufferRole::Write,
      },
      rund::AccelRunBinding{
          .buffer = &resources.write_values,
          .role = rund::kernel::BufferRole::Write,
      },
  };
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
    return false;
  }

  std::array<rund::kernel::u32, 8u> downloaded_keys{};
  std::array<rund::kernel::u32, 8u> downloaded_values{};
  const rund::AccelCheck key_download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write_keys, downloaded_keys.data(),
      downloaded_keys.size() * sizeof(rund::kernel::u32));
  const rund::AccelCheck value_download =
      rund::node::accel::DownloadAccelBuffer(
          resources.context, resources.write_values, downloaded_values.data(),
          downloaded_values.size() * sizeof(rund::kernel::u32));
  return key_download.ok && value_download.ok &&
         p::HashValues(downloaded_keys.data(), downloaded_keys.size()) ==
             p::HashValues(expected.keys.data(), expected.keys.size()) &&
         p::HashValues(downloaded_values.data(), downloaded_values.size()) ==
             p::HashValues(expected.values.data(), expected.values.size()) &&
         sort_identity::StableEqualKeyOrderValid(downloaded_keys,
                                                 downloaded_values);
}

} // namespace node_accel_contract
