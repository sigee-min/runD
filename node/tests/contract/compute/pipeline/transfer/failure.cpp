#include "local.hpp"

#include "model.hpp"

#include <array>
#include <cstdint>
#include <mutex>

namespace rund_node_test_pipeline_transfer {
namespace {

using namespace rund::compute;

[[nodiscard]] bool ClaimFailureRetriesWithoutPoison() {
  Fixture fixture{};
  TransferProbe probe{};
  active_probe = &probe;
  constexpr std::array<std::uint32_t, 4u> first{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 3u> tail{5u, 6u, 7u};
  const auto frames = uploads(fixture, first, tail);
  {
    std::lock_guard lock{fixture.device->claims->gate};
    fixture.inputs[0u]->writer = true;
  }
  const PipelineFrameUploadResult rejected = transfer_locked(fixture, frames);
  {
    std::lock_guard lock{fixture.device->claims->gate};
    fixture.inputs[0u]->writer = false;
  }
  const PipelineFrameUploadResult retried = transfer_locked(fixture, frames);
  active_probe = nullptr;
  return rejected.transfer.status.reason() == Reason::BufferBusy &&
         rejected.bytes == 0u && rejected.transfer.command_submits == 0u &&
         retried.transfer.status && retried.bytes == Fixture::total_bytes &&
         retried.transfer.command_submits == 1u && probe.upload_calls == 1u &&
         fixture.inputs[0u]->generation == 1u &&
         fixture.inputs[1u]->generation == 1u &&
         !fixture.inputs[0u]->poisoned && !fixture.inputs[1u]->poisoned;
}

[[nodiscard]] bool UploadFailurePoisonsOneAttempt() {
  Fixture fixture{};
  TransferProbe probe{.fail_upload = true};
  active_probe = &probe;
  constexpr std::array<std::uint32_t, 4u> first{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 3u> tail{5u, 6u, 7u};
  const PipelineFrameUploadResult result =
      transfer_locked(fixture, uploads(fixture, first, tail));
  active_probe = nullptr;
  return result.transfer.status.reason() == Reason::TransferInvalid &&
         result.bytes == 0u && result.transfer.command_submits == 1u &&
         probe.upload_calls == 1u && fixture.inputs[0u]->generation == 0u &&
         fixture.inputs[1u]->generation == 0u && fixture.inputs[0u]->poisoned &&
         fixture.inputs[1u]->poisoned &&
         fixture.state->stats.transfer_submissions.host_to_device ==
             result.transfer.command_submits &&
         fixture.state->stats.uploaded_bytes == 0u;
}

} // namespace

bool CheckBatchClaimAndFailure() {
  return ClaimFailureRetriesWithoutPoison() && UploadFailurePoisonsOneAttempt();
}

} // namespace rund_node_test_pipeline_transfer
