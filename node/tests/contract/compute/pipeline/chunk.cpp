#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/accel/context/transfer.hpp"
#include "src/compute/device/state.hpp"

#include <memory>
#include <vector>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckVulkanCheckpointChunking(rund::compute::Device &device,
                                                const Backend backend) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  static_cast<void>(device);
  static_cast<void>(backend);
  return 0;
#else
  using namespace rund::compute;
  if (backend != Backend::Vulkan) {
    return 0;
  }
  constexpr std::size_t staging_budget = 1024u * 1024u;
  constexpr std::size_t field_bytes = staging_budget * 3u / 4u;
  constexpr std::size_t field_count = field_bytes / sizeof(std::int32_t);
  std::vector<std::int32_t> initial(field_count, 7);
  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-checkpoint-chunking", field_count,
                             [](auto value) { return value + 1; })
          .compile();
  auto left_first = device.upload<std::int32_t>(initial);
  auto left_second = device.buffer<std::int32_t>(field_count);
  auto right_first = device.upload<std::int32_t>(initial);
  auto right_second = device.buffer<std::int32_t>(field_count);
  if (!advance || !left_first || !left_second || !right_first ||
      !right_second) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .state(*left_first, *left_second)
                      .state(*right_first, *right_second)
                      .then(*advance, read(*left_first), write(*left_second))
                      .then(*advance, read(*right_first), write(*right_second))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 2;
  }
  const Stats before_snapshot = prepared->stats();
  const auto saved = prepared->snapshot();
  const Stats after_snapshot = prepared->stats();
  if (!saved ||
      after_snapshot.publication.snapshot_byte_count != field_bytes * 2u ||
      after_snapshot.command_submits != before_snapshot.command_submits + 2u ||
      after_snapshot.downloaded_bytes !=
          before_snapshot.downloaded_bytes + field_bytes * 2u) {
    return 3;
  }

  auto restored_left_first = device.buffer<std::int32_t>(field_count);
  auto restored_left_second = device.buffer<std::int32_t>(field_count);
  auto restored_right_first = device.buffer<std::int32_t>(field_count);
  auto restored_right_second = device.buffer<std::int32_t>(field_count);
  if (!restored_left_first || !restored_left_second || !restored_right_first ||
      !restored_right_second) {
    return 4;
  }
  auto restored = pipeline(device)
                      .state(*restored_left_first, *restored_left_second)
                      .state(*restored_right_first, *restored_right_second)
                      .then(*advance, read(*restored_left_first),
                            write(*restored_left_second))
                      .then(*advance, read(*restored_right_first),
                            write(*restored_right_second))
                      .restore(*saved)
                      .commit()
                      .prepare();
  if (!restored) {
    return 5;
  }
  const Stats restored_stats = restored->stats();
  const MemoryStats restored_memory = restored->memory();
  if (restored_stats.command_submits != 4u ||
      restored_stats.uploaded_bytes != field_bytes * 4u ||
      restored_stats.publication.restore_byte_count != field_bytes * 4u ||
      restored_memory.staging.cumulative < field_bytes * 4u ||
      restored_memory.staging.peak < field_bytes ||
      restored_memory.staging.peak >
          restored_memory.staging.current + staging_budget ||
      restored_memory.staging.budget < staging_budget) {
    return 6;
  }

  constexpr std::array<std::uint32_t, 1u> word{0x44332211u};
  auto overlapping = Upload(device, word);
  const std::shared_ptr<detail::DeviceState> &device_state =
      detail::DeviceAccess::state(device);
  detail::AccelDeviceState *const accel =
      device_state == nullptr ? nullptr : detail::accel_device(*device_state);
  const std::shared_ptr<detail::BufferState> buffer_state =
      overlapping ? detail::BufferAccess::state(*overlapping) : nullptr;
  detail::AccelBufferState *const native =
      buffer_state == nullptr ? nullptr : detail::accel_buffer(*buffer_state);
  std::array<std::byte, sizeof(std::uint32_t)> expected{};
  if (accel == nullptr || native == nullptr ||
      !rund::node::accel::DownloadAccelBuffer(accel->context, native->buffer,
                                              expected.data(), expected.size())
           .ok) {
    return 7;
  }
  const std::byte first_byte{0xAAu};
  const std::byte second_byte{0xBBu};
  const std::array<rund::node::accel::detail::UploadEntry, 2u> overlap_requests{
      {
          {.buffer = &native->buffer,
           .data = &first_byte,
           .bytes = 1u,
           .offset = 1u},
          {.buffer = &native->buffer,
           .data = &second_byte,
           .bytes = 1u,
           .offset = 2u},
      }};
  std::array<rund::node::accel::detail::UploadRoute, 2u> overlap_routes{};
  const auto overlap_transfer = rund::node::accel::detail::UploadAccelBuffers(
      accel->context, overlap_requests, overlap_routes);
  expected[1] = first_byte;
  expected[2] = second_byte;
  std::array<std::byte, sizeof(std::uint32_t)> observed{};
  if (!overlap_transfer.check.ok ||
      overlap_transfer.staging_bytes != sizeof(std::uint32_t) * 2u ||
      overlap_transfer.staging_peak_bytes != sizeof(std::uint32_t) ||
      overlap_transfer.buffer_allocations + overlap_transfer.buffer_reuses !=
          2u ||
      overlap_transfer.command_submits != 4u ||
      !rund::node::accel::DownloadAccelBuffer(accel->context, native->buffer,
                                              observed.data(), observed.size())
           .ok ||
      observed != expected) {
    return 8;
  }
  return 0;
#endif
}

} // namespace rund_node_test_pipeline
