#include "local.hpp"

#include <rund/compute/async.hpp>

#include "../../../../../src/compute/open/probe.hpp"

#include <utility>

namespace rund_node_flow_contract {

int CheckDevice() {
  using namespace rund::compute;
  auto opened = open(Target::cpu(1u));
  if (!opened) {
    return 1;
  }
  Device live = std::move(*opened);
  if (!live || opened->valid()) {
    return 2;
  }
  const auto moved_backend = opened->backend();
  if (moved_backend || moved_backend.reason() != Reason::DeviceInvalid) {
    return 3;
  }

  std::uint64_t open_count = 0u;
  const auto rejected = [&] {
    detail::ScopedOpenProbe probe{open_count};
    return on(*opened)
        .map<std::uint32_t>("moved-device", 1u,
                            [](auto value) { return value; })
        .compile();
  }();
  if (rejected || rejected.reason() != Reason::DeviceInvalid ||
      open_count != 0u) {
    return 4;
  }

  std::uint64_t async_open_count = 0u;
  const auto async_rejected = [&] {
    detail::ScopedOpenProbe probe{async_open_count};
    return on(*opened)
        .map<std::uint32_t>("moved-device-async", 1u,
                            [](auto value) { return value; })
        .compile_async();
  }();
  if (async_rejected || async_rejected.reason() != Reason::DeviceInvalid ||
      async_open_count != 0u) {
    return 5;
  }
  const auto live_backend = live.backend();
  return live_backend && *live_backend == Backend::Cpu ? 0 : 6;
}

} // namespace rund_node_flow_contract
