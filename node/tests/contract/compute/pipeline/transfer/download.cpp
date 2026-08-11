#include "local.hpp"

#include "model.hpp"

#include <array>
#include <cstddef>

namespace rund_node_test_pipeline_transfer {

bool CheckBatchDownloadAndHash() {
  using namespace rund::compute;

  Fixture failed{};
  TransferProbe failed_probe{.fail_download_once = true};
  active_probe = &failed_probe;
  std::array<std::byte, Fixture::first_bytes> first_result{};
  std::array<std::byte, Fixture::tail_bytes> tail_result{};
  const std::array<PipelineFrameDownload, 2u> forward{{
      {.buffer = failed.outputs[0u].get(),
       .data = first_result.data(),
       .bytes = first_result.size(),
       .output = 0u},
      {.buffer = failed.outputs[1u].get(),
       .data = tail_result.data(),
       .bytes = tail_result.size(),
       .output = 1u},
  }};
  const PipelineFrameDownloadResult rejected = transfer_locked(failed, forward);
  const PipelineFrameDownloadResult retried = transfer_locked(failed, forward);
  const std::uint64_t forward_hash = failed.state->stats.output_hash;
  if (rejected.transfer.status.reason() != Reason::TransferInvalid ||
      rejected.bytes != 0u || rejected.events != 0u ||
      rejected.transfer.command_submits != 1u ||
      rejected.transfer.readback_ns != 0u || !retried.transfer.status ||
      retried.bytes != Fixture::total_bytes || retried.events != 2u ||
      retried.transfer.command_submits != 1u ||
      retried.transfer.readback_ns != 17u ||
      failed_probe.download_calls != 2u ||
      failed.state->stats.transfer_submissions.device_to_host !=
          rejected.transfer.command_submits +
              retried.transfer.command_submits ||
      failed.state->stats.downloaded_bytes != Fixture::total_bytes ||
      failed.state->stats.download_events != 2u ||
      failed.state->stats.readback_ns != retried.transfer.readback_ns ||
      forward_hash == 0u) {
    active_probe = nullptr;
    return false;
  }

  Fixture reversed{};
  TransferProbe reversed_probe{};
  active_probe = &reversed_probe;
  std::array<std::byte, Fixture::first_bytes> reversed_first{};
  std::array<std::byte, Fixture::tail_bytes> reversed_tail{};
  const std::array<PipelineFrameDownload, 2u> reverse{{
      {.buffer = reversed.outputs[1u].get(),
       .data = reversed_tail.data(),
       .bytes = reversed_tail.size(),
       .output = 1u},
      {.buffer = reversed.outputs[0u].get(),
       .data = reversed_first.data(),
       .bytes = reversed_first.size(),
       .output = 0u},
  }};
  const PipelineFrameDownloadResult ordered =
      transfer_locked(reversed, reverse);
  active_probe = nullptr;
  return ordered.transfer.status && ordered.events == 2u &&
         ordered.transfer.command_submits == 1u &&
         reversed_probe.download_calls == 1u &&
         reversed.state->stats.output_hash == forward_hash;
}

} // namespace rund_node_test_pipeline_transfer
