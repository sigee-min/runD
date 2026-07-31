#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "src/accel/kernel/prepared.hpp"

#include "bindings.hpp"

namespace node_accel_contract::gather {

template <typename T>
[[nodiscard]] bool MatchesReference(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::GatherElement element, const std::array<T, 6u> &values,
    const std::array<rund::kernel::u32, 4u> &indices) {
  namespace fix = node_accel_contract::primitive;
  namespace run = node_accel_contract::gather::match;

  if (!pick.check.ok) {
    return false;
  }
  const run::Reference<T> ref = run::BuildReference(element, values, indices);
  if (!ref.ok) {
    return false;
  }

  run::Resources<T> resources =
      run::BuildResources(pick, scalar, element, values, indices);
  if (!resources.kernel.check.ok) {
    return false;
  }
  const auto bindings = run::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = indices.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return false;
  }

  std::array<T, 4u> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, downloaded.data(),
      downloaded.size() * sizeof(T));
  return download.ok &&
         fix::HashValues(downloaded.data(), downloaded.size()) ==
             fix::HashValues(ref.expected.data(), ref.expected.size());
}

bool MatchesU32(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u32>(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::GatherElement::U32,
      std::array<rund::kernel::u32, 6u>{10u, 20u, 30u, 40u, 50u, 60u},
      std::array<rund::kernel::u32, 4u>{3u, 0u, 3u, 5u});
}

bool MatchesU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::GatherElement::U64,
      std::array<rund::kernel::u64, 6u>{7u, 11u, 13u, 17u, 19u, 23u},
      std::array<rund::kernel::u32, 4u>{4u, 1u, 2u, 0u});
}

bool PreparedRetainsStorage(const rund::AccelDevice &pick) {
  namespace run = node_accel_contract::gather::match;
  constexpr std::array<rund::kernel::u32, 6u> values{10u, 20u, 30u,
                                                     40u, 50u, 60u};
  constexpr std::array<rund::kernel::u32, 4u> indices{3u, 0u, 3u, 5u};
  run::Resources<rund::kernel::u32> resources =
      run::BuildResources(pick, rund::kernel::ComputeScalar::Lane32,
                          rund::kernel::GatherElement::U32, values, indices);
  if (!resources.kernel.check.ok) {
    return false;
  }
  const auto bindings = run::Bindings(resources);
  const rund::node::accel::detail::PreparedKernelRun prepared =
      rund::node::accel::detail::PrepareKernelRun(
          resources.context, resources.kernel,
          rund::AccelRun{
              .bindings = bindings.data(),
              .binding_count = bindings.size(),
              .tile_count = indices.size(),
              .fresh_evidence = true,
          });
  if (!prepared.ok) {
    return false;
  }
  resources.source = {};
  resources.index = {};
  resources.output = {};
  const rund::AccelEvidence evidence =
      rund::node::accel::detail::RunPreparedKernel(resources.context, prepared);
  return evidence.ok && evidence.host_to_device_bytes == 0u &&
         evidence.device_to_host_bytes == 0u;
}

} // namespace node_accel_contract::gather
