#include "actual.hpp"

#include "actual/internal.hpp"
#include "actual_binary.hpp"

#include "src/compute/device/state.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency::device_vsm_test {

bool CheckActualDeviceVsm(const rund::compute::Backend backend,
                          const ActualPrepare prepare,
                          const ActualQueueCount queue_count) noexcept {
  auto opened = rund::compute::open(TargetFor(backend));
  if (!opened) {
    std::fprintf(stderr, "DeviceVsm actual backend=%u open=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(opened.reason()));
    return false;
  }
  rund::compute::Device device = std::move(opened).value();
  const std::shared_ptr<rund::compute::detail::DeviceState> state =
      rund::compute::detail::DeviceAccess::state(device);
  std::array<std::uint64_t, 4u> retained{};
  std::size_t index = 0u;
  for (const std::uint64_t pages : {5u, 9u, 257u, 100000u}) {
    if (!RunActual(backend, state, pages, prepare, queue_count,
                   retained[index++])) {
      return false;
    }
  }
  const bool fixed = retained[0u] != 0u && retained[0u] == retained[1u] &&
                     retained[0u] == retained[2u] &&
                     retained[0u] == retained[3u];
  if (!fixed) {
    std::fprintf(stderr,
                 "DeviceVsm retained backend=%u bytes=%llu/%llu/%llu/%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(retained[0u]),
                 static_cast<unsigned long long>(retained[1u]),
                 static_cast<unsigned long long>(retained[2u]),
                 static_cast<unsigned long long>(retained[3u]));
  }
  return fixed &&
         CheckActualBinaryDeviceVsm(backend, state, prepare, queue_count);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
