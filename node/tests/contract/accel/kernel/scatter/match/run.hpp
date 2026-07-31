#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <node/accel/context.hpp>

#include "repeat.hpp"

namespace node_accel_contract::scatter {

template <typename T>
[[nodiscard]] bool MatchesReference(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ScatterElement element, const std::array<T, 4u> &values,
    const std::array<rund::kernel::u32, 4u> &indices,
    const std::array<T, 6u> &initial_output) {
  namespace fix = node_accel_contract::primitive;
  namespace run = node_accel_contract::scatter::match;

  if (!pick.check.ok) {
    return false;
  }
  const run::Reference<T> ref =
      run::BuildReference(element, values, indices, initial_output);
  if (!ref.ok) {
    return false;
  }

  run::Resources<T> resources = run::BuildResources(
      pick, scalar, element, values, indices, initial_output);
  if (!resources.kernel.check.ok) {
    return false;
  }
  const run::RepeatedRun repeated = run::RunTwice(resources, values.size());
  if (!repeated.first.ok || !repeated.second.ok ||
      repeated.first.host_to_device_bytes != 0u ||
      repeated.first.device_to_host_bytes != 0u ||
      repeated.second.host_to_device_bytes != 0u ||
      repeated.second.device_to_host_bytes != 0u) {
    return false;
  }

  std::array<T, 6u> downloaded{};
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
      rund::kernel::ScatterElement::U32,
      std::array<rund::kernel::u32, 4u>{10u, 20u, 30u, 40u},
      std::array<rund::kernel::u32, 4u>{3u, 0u, 5u, 2u},
      std::array<rund::kernel::u32, 6u>{1u, 2u, 3u, 4u, 5u, 6u});
}

bool MatchesU64(const rund::AccelDevice &pick) {
  return MatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ScatterElement::U64,
      std::array<rund::kernel::u64, 4u>{7u, 11u, 13u, 17u},
      std::array<rund::kernel::u32, 4u>{1u, 4u, 0u, 3u},
      std::array<rund::kernel::u64, 6u>{101u, 102u, 103u, 104u, 105u, 106u});
}

} // namespace node_accel_contract::scatter
