#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckReusableCheckpoint(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 2u> initial{4, 9};
  auto advance =
      rund::compute::on(device)
          .map<std::int32_t>("pipeline-installed-checkpoint", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto published = device.upload<std::int32_t>(initial);
  auto pending = device.buffer<std::int32_t>(initial.size());
  if (!advance || !published || !pending) {
    return 1;
  }
  auto prepared = rund::compute::pipeline(device)
                      .state(*published, *pending)
                      .then(*advance, rund::compute::read(*published),
                            rund::compute::write(*pending))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 2;
  }
  auto latest_result = prepared->latest_device_state();
  auto storage_result = prepared->snapshot_storage();
  if (!latest_result || !storage_result) {
    return 3;
  }
  rund::compute::LatestDeviceState latest = *latest_result;
  rund::compute::SnapshotStorage storage = std::move(storage_result).value();
  if (!prepared->snapshot_into(storage) || !storage.has_snapshot() ||
      storage.generation() != 0u || storage.capacity() != sizeof(initial) ||
      prepared->checkpoint_stats().device_state_acquire_count != 1u) {
    return 4;
  }
  if (!prepared->run() || latest.generation() != 1u) {
    return 5;
  }

  auto resumed = rund::compute::pipeline(device)
                     .state(*published, *pending)
                     .then(*advance, rund::compute::read(*published),
                           rund::compute::write(*pending))
                     .restore(latest)
                     .commit()
                     .prepare();
  if (!resumed || resumed->generation() != 1u ||
      resumed->checkpoint_stats().device_state_rebase_count != 1u ||
      resumed->checkpoint_stats().device_state_copy_byte_count != 0u ||
      !resumed->snapshot_into(storage) || storage.generation() != 1u) {
    return 6;
  }

  auto restored_first = device.buffer<std::int32_t>(initial.size());
  auto restored_second = device.buffer<std::int32_t>(initial.size());
  if (!restored_first || !restored_second) {
    return 7;
  }
  auto restored = rund::compute::pipeline(device)
                      .state(*restored_first, *restored_second)
                      .then(*advance, rund::compute::read(*restored_first),
                            rund::compute::write(*restored_second))
                      .restore(storage)
                      .commit()
                      .prepare();
  if (!restored || restored->generation() != 1u || !restored->run()) {
    return 8;
  }
  std::array<std::int32_t, initial.size()> observed{};
  const auto read = restored->read(*restored_second, observed);
  return read && observed == std::array<std::int32_t, 2u>{6, 11} ? 0 : 9;
}

} // namespace rund::package_example::pipeline
