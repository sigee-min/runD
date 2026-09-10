#include "actual_binary.hpp"

#include "actual_binary/internal.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test {

bool CheckActualBinaryDeviceVsm(
    const rund::compute::Backend backend,
    const std::shared_ptr<rund::compute::detail::DeviceState> &device,
    const ActualPrepare prepare, const ActualQueueCount queue_count) noexcept {
  std::array<std::uint64_t, 3u> retained{};
  std::size_t index = 0u;
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    if (!binary_detail::RunBinary(backend, device, pages, prepare, queue_count,
                                  retained[index++])) {
      return false;
    }
  }
  return retained[0u] != 0u && retained[0u] == retained[1u] &&
         retained[0u] == retained[2u];
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
