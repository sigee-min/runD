#include "local.hpp"

#include "../../allocation.hpp"
#include "model.hpp"

#include <rund/compute/pipeline.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace rund_node_test_pipeline_transfer {

namespace {

using namespace rund::compute;
using rund::node::accel::detail::DownloadRangeState;
using rund::node::accel::detail::TransferAuthority;

[[nodiscard]] bool ResidentDisjoint() {
  ResidentBatchFixture fixture = make_resident_batch_spy();
  std::array<std::byte, 8u> first{};
  std::array<std::byte, 12u> second{};
  const std::array<VirtualRead, 2u> ranges{{
      {.offset = 4u, .bytes = first},
      {.offset = 20u, .bytes = second},
  }};
  const Status status = fixture.backing->read_batch(ranges);
  constexpr std::array<std::byte, 8u> expected_first{
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08},
      std::byte{0x09}, std::byte{0x0a}, std::byte{0x0b}, std::byte{0x0c}};
  constexpr std::array<std::byte, 12u> expected_second{
      std::byte{0x15}, std::byte{0x16}, std::byte{0x17}, std::byte{0x18},
      std::byte{0x19}, std::byte{0x1a}, std::byte{0x1b}, std::byte{0x1c},
      std::byte{0x1d}, std::byte{0x1e}, std::byte{0x1f}, std::byte{0x20}};
  return status && fixture.spy->batch_calls == 1u &&
         fixture.spy->batch_routes == 2u &&
         fixture.spy->authority == TransferAuthority::Shared &&
         fixture.spy->outcomes[0u].state == DownloadRangeState::Complete &&
         fixture.spy->outcomes[0u].confirmed_bytes == first.size() &&
         fixture.spy->outcomes[0u].hash_valid &&
         fixture.spy->outcomes[0u].payload_hash == 0x4cc23a8486396823ull &&
         fixture.spy->outcomes[1u].state == DownloadRangeState::Complete &&
         fixture.spy->outcomes[1u].confirmed_bytes == second.size() &&
         fixture.spy->outcomes[1u].hash_valid &&
         fixture.spy->outcomes[1u].payload_hash == 0xe11badb1b2ac544bull &&
         std::memcmp(first.data(), expected_first.data(), first.size()) == 0 &&
         std::memcmp(second.data(), expected_second.data(), second.size()) == 0;
}

[[nodiscard]] bool ResidentScalarFallback() {
  ResidentBatchFixture fixture = make_resident_scalar_spy();
  std::array<std::byte, 8u> first{};
  std::array<std::byte, 8u> second{};
  second.fill(std::byte{0x7f});
  const std::array<VirtualRead, 2u> ranges{{
      {.offset = 0u, .bytes = first},
      {.offset = 60u, .bytes = second},
  }};
  const Status status = fixture.backing->read_batch(ranges);
  constexpr std::array<std::byte, 8u> expected_first{
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};
  return !status && status.reason() == Reason::ShapeMismatch &&
         fixture.spy->batch_calls == 0u && fixture.spy->scalar_calls == 1u &&
         fixture.spy->scalar_routes == 1u &&
         std::memcmp(first.data(), expected_first.data(), first.size()) == 0 &&
         std::all_of(second.begin(), second.end(), [](const std::byte value) {
           return value == std::byte{0x7f};
         });
}

[[nodiscard]] bool ResidentLateFailure() {
  ResidentBatchFixture fixture = make_resident_batch_spy();
  fixture.spy->late_failure = true;
  std::array<std::byte, 8u> first{};
  std::array<std::byte, 8u> second{};
  std::array<std::byte, 8u> third{};
  second.fill(std::byte{0x6d});
  third.fill(std::byte{0x7e});
  const std::array<VirtualRead, 3u> ranges{{
      {.offset = 0u, .bytes = first},
      {.offset = 16u, .bytes = second},
      {.offset = 32u, .bytes = third},
  }};
  const Status status = fixture.backing->read_batch(ranges);
  constexpr std::array<std::byte, 8u> expected_first{
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};
  return !status && status.reason() == Reason::TransferInvalid &&
         fixture.spy->batch_calls == 1u && fixture.spy->batch_routes == 3u &&
         fixture.spy->scalar_calls == 0u &&
         fixture.spy->outcomes[0u].state == DownloadRangeState::Complete &&
         fixture.spy->outcomes[0u].confirmed_bytes == first.size() &&
         fixture.spy->outcomes[0u].hash_valid &&
         fixture.spy->outcomes[1u].state ==
             DownloadRangeState::FailedMayWrite &&
         fixture.spy->outcomes[1u].confirmed_bytes == 2u &&
         !fixture.spy->outcomes[1u].hash_valid &&
         fixture.spy->outcomes[2u].state == DownloadRangeState::Untouched &&
         fixture.spy->outcomes[2u].confirmed_bytes == 0u &&
         std::memcmp(first.data(), expected_first.data(), first.size()) == 0 &&
         second[0u] == std::byte{0x11} && second[1u] == std::byte{0x12} &&
         std::all_of(
             second.begin() + 2u, second.end(),
             [](const std::byte value) { return value == std::byte{0x6d}; }) &&
         std::all_of(third.begin(), third.end(), [](const std::byte value) {
           return value == std::byte{0x7e};
         });
}

} // namespace

bool CheckBatchSuccessAndAccounting() {
  using namespace rund::compute;
  using rund::compute::detail::pipeline_memory;

  Fixture fixture{};
  TransferProbe probe{};
  active_probe = &probe;
  constexpr std::array<std::uint32_t, 4u> first{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 3u> tail{9u, 8u, 7u};
  const auto upload_frames = uploads(fixture, first, tail);
  node_compute_allocation::Start();
  const PipelineFrameUploadResult uploaded =
      transfer_locked(fixture, upload_frames);
  node_compute_allocation::Stop();
  const std::uint64_t upload_allocations = node_compute_allocation::Count();
  if (!uploaded.transfer.status || uploaded.bytes != Fixture::total_bytes ||
      uploaded.transfer.command_submits != 1u || probe.upload_calls != 1u ||
      probe.upload_routes != 2u || upload_allocations != 0u ||
      fixture.inputs[0u]->generation != 1u ||
      fixture.inputs[1u]->generation != 1u || fixture.inputs[0u]->poisoned ||
      fixture.inputs[1u]->poisoned ||
      std::memcmp(native(*fixture.inputs[0u])->bytes.data(), first.data(),
                  Fixture::first_bytes) != 0 ||
      std::memcmp(native(*fixture.inputs[1u])->bytes.data(), tail.data(),
                  Fixture::tail_bytes) != 0) {
    active_probe = nullptr;
    return false;
  }
  const Stats upload_stats = fixture.state->stats;
  const MemoryStats upload_memory = pipeline_memory(fixture.state);
  if (upload_stats.transfer_submissions.host_to_device !=
          uploaded.transfer.command_submits ||
      upload_stats.uploaded_bytes != uploaded.bytes ||
      upload_stats.host_write_bytes != uploaded.bytes ||
      upload_stats.buffer_allocations != uploaded.transfer.buffer_allocations ||
      upload_stats.buffer_reuses != uploaded.transfer.buffer_reuses ||
      upload_memory.transfer.cumulative != uploaded.bytes ||
      upload_memory.transfer.peak != uploaded.bytes ||
      upload_memory.staging.cumulative != uploaded.transfer.staging_bytes ||
      upload_memory.staging.reused != uploaded.transfer.staging_reused_bytes ||
      upload_memory.staging.peak != uploaded.transfer.staging_peak_bytes ||
      upload_memory.staging.budget != uploaded.transfer.staging_budget) {
    active_probe = nullptr;
    return false;
  }
  {
    // Model Pipeline::run opening its next Stats epoch. The retained upload
    // result remains complete physical evidence after this reset.
    std::lock_guard lock{fixture.state->gate};
    fixture.state->stats = {};
  }

  std::array<std::byte, Fixture::first_bytes> first_result{};
  std::array<std::byte, Fixture::tail_bytes> tail_result{};
  const std::array<PipelineFrameDownload, 2u> download_frames{{
      {.buffer = fixture.outputs[0u].get(),
       .data = first_result.data(),
       .bytes = first_result.size(),
       .output = 0u},
      {.buffer = fixture.outputs[1u].get(),
       .data = tail_result.data(),
       .bytes = tail_result.size(),
       .output = 1u},
  }};
  node_compute_allocation::Start();
  const PipelineFrameDownloadResult downloaded =
      transfer_locked(fixture, download_frames);
  node_compute_allocation::Stop();
  const std::uint64_t download_allocations = node_compute_allocation::Count();
  const Stats &stats = fixture.state->stats;
  const MemoryStats memory = pipeline_memory(fixture.state);
  active_probe = nullptr;
  return downloaded.transfer.status &&
         downloaded.bytes == Fixture::total_bytes && downloaded.events == 2u &&
         downloaded.transfer.command_submits == 1u &&
         downloaded.transfer.readback_ns == 17u && probe.download_calls == 1u &&
         probe.download_routes == 2u && download_allocations == 0u &&
         fixture.state->unobserved_outputs == 0u &&
         fixture.state->outputs[0u].observed &&
         fixture.state->outputs[1u].observed && stats.output_hash != 0u &&
         stats.transfer_submissions.host_to_device == 0u &&
         stats.transfer_submissions.device_to_host ==
             downloaded.transfer.command_submits &&
         stats.uploaded_bytes == 0u && stats.host_write_bytes == 0u &&
         stats.downloaded_bytes == downloaded.bytes &&
         stats.download_events == downloaded.events &&
         stats.buffer_allocations == downloaded.transfer.buffer_allocations &&
         stats.buffer_reuses == downloaded.transfer.buffer_reuses &&
         stats.readback_ns == downloaded.transfer.readback_ns &&
         stats.readback_ns == 17u &&
         fixture.state->transfer_bytes == uploaded.bytes + downloaded.bytes &&
         fixture.state->transfer_peak == Fixture::total_bytes &&
         fixture.state->staging_bytes ==
             uploaded.transfer.staging_bytes +
                 downloaded.transfer.staging_bytes &&
         fixture.state->staging_reused ==
             uploaded.transfer.staging_reused_bytes +
                 downloaded.transfer.staging_reused_bytes &&
         fixture.state->staging_peak == uploaded.transfer.staging_peak_bytes &&
         fixture.state->staging_budget == uploaded.transfer.staging_budget &&
         memory.available() &&
         memory.transfer.cumulative == uploaded.bytes + downloaded.bytes &&
         memory.transfer.peak == Fixture::total_bytes &&
         memory.staging.cumulative == fixture.state->staging_bytes &&
         memory.staging.reused == fixture.state->staging_reused &&
         memory.staging.peak == fixture.state->staging_peak &&
         memory.staging.budget == fixture.state->staging_budget &&
         std::memcmp(first_result.data(),
                     native(*fixture.outputs[0u])->bytes.data(),
                     first_result.size()) == 0 &&
         std::memcmp(tail_result.data(),
                     native(*fixture.outputs[1u])->bytes.data(),
                     tail_result.size()) == 0 &&
         ResidentDisjoint() && ResidentScalarFallback() &&
         ResidentLateFailure();
}

} // namespace rund_node_test_pipeline_transfer
