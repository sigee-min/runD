#pragma once

#include "src/accel/backend/catalog.hpp"
#include "src/accel/cpu/ops.hpp"

#include <accel/device.hpp>

namespace node_accel_contract {

[[nodiscard]] inline rund::AccelDevice
PickCpu(const rund::AccelPolicy &policy) {
  const rund::node::accel::detail::BackendEntry catalog[]{
      rund::node::accel::detail::CpuEntry()};
  return rund::node::accel::detail::PickFromCatalog(catalog, policy);
}

} // namespace node_accel_contract
