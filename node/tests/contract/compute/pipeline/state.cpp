#include "local.hpp"

#include "../allocation.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/claim.hpp"
#include "src/compute/pipeline/state.hpp"

#include <memory>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckTransactionalGenerations(
    rund::compute::Device &device, const rund::compute::Backend backend,
    std::uint64_t &state_hash, std::uint64_t &mixed_hash) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> initial{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> once{2, 3, 4, 5};
  constexpr std::array<std::int32_t, 4u> twice{3, 4, 5, 6};
  constexpr std::array<std::int32_t, 4u> thrice{4, 5, 6, 7};
  constexpr std::array<std::int32_t, 4u> four{5, 6, 7, 8};
  constexpr std::array<std::int32_t, 4u> five{6, 7, 8, 9};
  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-generation-advance", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = Upload(device, initial);
  auto second = device.buffer<std::int32_t>(initial.size());
  if (!advance || !first || !second) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .state(*first, *second)
                      .then(*advance, read(*first), write(*second))
                      .commit()
                      .prepare();
  std::array<std::int32_t, initial.size()> observed{};
  if (!prepared || !prepared->run() || prepared->generation() != 1u ||
      !ReadExact(*prepared, *second, observed) || observed != once ||
      prepared->stats().output_hash == 0u || !prepared->run() ||
      prepared->generation() != 2u || !ReadExact(*prepared, *first, observed) ||
      observed != twice || !prepared->run() || prepared->generation() != 3u ||
      !ReadExact(*prepared, *second, observed) || observed != thrice) {
    return 2;
  }
  const std::uint64_t current_state_hash = prepared->stats().output_hash;
  if (state_hash == 0u) {
    state_hash = current_state_hash;
  } else if (state_hash != current_state_hash) {
    return 14;
  }
  if (backend == Backend::Cpu) {
    node_compute_allocation::Start();
  }
  auto odd_result = prepared->snapshot();
  if (backend == Backend::Cpu) {
    node_compute_allocation::Stop();
  }
  if (!odd_result || odd_result->generation() != 3u || !prepared->run() ||
      prepared->generation() != 4u || !ReadExact(*prepared, *first, observed) ||
      observed != four ||
      (backend == Backend::Cpu && node_compute_allocation::Count() != 3u)) {
    return 3;
  }
  auto even_result = prepared->snapshot();
  if (!even_result || even_result->generation() != 4u) {
    return 4;
  }
  const auto verify_restore = [&](const StateSnapshot &saved,
                                  const std::uint64_t generation,
                                  const std::array<std::int32_t, 4u> expected) {
    auto restored_first = device.buffer<std::int32_t>(initial.size());
    auto restored_second = device.buffer<std::int32_t>(initial.size());
    if (!restored_first || !restored_second) {
      return false;
    }
    auto restored =
        pipeline(device)
            .state(*restored_first, *restored_second)
            .then(*advance, read(*restored_first), write(*restored_second))
            .restore(saved)
            .commit()
            .prepare();
    const Stats restored_stats = restored ? restored->stats() : Stats{};
    Status repeated_restore = Status::success();
    if (backend == Backend::Cpu && restored) {
      node_compute_allocation::Start();
      repeated_restore = restored->restore(saved);
      node_compute_allocation::Stop();
    }
    const std::uint64_t expected_restore_bytes =
        backend == Backend::Cpu ? 0u : sizeof(initial) * 2u;
    const std::uint64_t expected_restore_submits =
        backend == Backend::Vulkan ? 1u : 0u;
    return restored && repeated_restore &&
           (backend != Backend::Cpu ||
            node_compute_allocation::Count() == 0u) &&
           restored_stats.uploaded_bytes == expected_restore_bytes &&
           restored_stats.command_submits == expected_restore_submits &&
           restored_stats.publication.restore_byte_count ==
               sizeof(initial) * 2u &&
           restored->generation() == generation && restored->run() &&
           restored->generation() == generation + 1u &&
           ReadExact(*restored, *restored_second, observed) &&
           observed == expected;
  };
  if (!verify_restore(*odd_result, 3u, four) ||
      !verify_restore(*even_result, 4u, five)) {
    return 5;
  }

  auto recurrent_first = Upload(device, initial);
  auto recurrent_second = device.buffer<std::int32_t>(initial.size());
  if (!recurrent_first || !recurrent_second) {
    return 51;
  }
  auto recurrent = pipeline(device)
                       .state(*recurrent_first, *recurrent_second)
                       .repeat<2u>(*advance, read(*recurrent_first),
                                   write_final(*recurrent_second))
                       .commit()
                       .prepare();
  const std::shared_ptr<detail::PipelineState> recurrent_state =
      recurrent ? detail::PipelineStateAccess::state(*recurrent)
                : std::shared_ptr<detail::PipelineState>{};
  if (recurrent_state == nullptr || recurrent_state->steps.size() != 2u ||
      recurrent_state->steps[0].job == nullptr ||
      recurrent_state->steps[1].job == nullptr ||
      recurrent_state->steps[0].alternate_job == nullptr ||
      recurrent_state->steps[1].alternate_job == nullptr ||
      recurrent_state->steps[0].job->workspace == nullptr ||
      recurrent_state->steps[0].job->workspace !=
          recurrent_state->steps[1].job->workspace ||
      recurrent_state->steps[0].job->workspace !=
          recurrent_state->steps[0].alternate_job->workspace ||
      recurrent_state->steps[0].job->workspace !=
          recurrent_state->steps[1].alternate_job->workspace) {
    return 52;
  }
  constexpr std::array<std::int32_t, 4u> recurrent_once{3, 4, 5, 6};
  constexpr std::array<std::int32_t, 4u> recurrent_twice{5, 6, 7, 8};
  if (!recurrent || !recurrent->run() || recurrent->generation() != 1u ||
      !ReadExact(*recurrent, *recurrent_second, observed) ||
      observed != recurrent_once) {
    return 52;
  }
  auto recurrent_snapshot = recurrent->snapshot();
  if (!recurrent_snapshot || !recurrent->run() ||
      !ReadExact(*recurrent, *recurrent_first, observed) ||
      observed != recurrent_twice) {
    return 53;
  }
  auto restored_first = device.buffer<std::int32_t>(initial.size());
  auto restored_second = device.buffer<std::int32_t>(initial.size());
  if (!restored_first || !restored_second) {
    return 54;
  }
  auto recurrent_restore = pipeline(device)
                               .state(*restored_first, *restored_second)
                               .repeat<2u>(*advance, read(*restored_first),
                                           write_final(*restored_second))
                               .restore(*recurrent_snapshot)
                               .commit()
                               .prepare();
  if (!recurrent_restore || !recurrent_restore->run() ||
      !ReadExact(*recurrent_restore, *restored_second, observed) ||
      observed != recurrent_twice) {
    return 55;
  }

  // A submitted semantic failure advances the selected physical control but
  // must not advance the public generation. Rebase that same parity stream so
  // the valid retry publishes exactly the next generation.
  constexpr std::array<std::int32_t, 4u> source_values{10, 20, 30, 40};
  constexpr std::array<std::uint32_t, 4u> valid_indices{0u, 1u, 2u, 3u};
  constexpr std::array<std::uint32_t, 4u> invalid_indices{5u, 4u, 2u, 3u};
  constexpr std::array<std::uint32_t, 4u> recovery_indices{3u, 2u, 1u, 0u};
  constexpr std::array<std::int32_t, 4u> recovered_values{40, 30, 20, 10};
  auto gather = on(device)
                    .input<std::int32_t>(source_values.size())
                    .zip_input<std::uint32_t>(valid_indices.size())
                    .branch([](auto values, auto indices) {
                      return values.gather(indices);
                    })
                    .compile();
  auto source = Upload(device, source_values);
  auto indices = Upload(device, valid_indices);
  auto pending = device.buffer<std::int32_t>(source_values.size());
  if (!gather || !source || !indices || !pending) {
    return 6;
  }
  auto retry = pipeline(device)
                   .state(*source, *pending)
                   .then(*gather, read(*source, *indices), write(*pending))
                   .commit()
                   .prepare();
  const auto retry_latest =
      retry ? retry->latest_device_state()
            : Result<LatestDeviceState>::fail(Reason::PipelineInvalid);
  const Status first_retry =
      retry ? retry->run() : Status::fail(Reason::PipelineInvalid);
  const bool first_read = retry && ReadExact(*retry, *pending, observed);
  const bool first_overwrite = Overwrite(*indices, invalid_indices);
  if (!retry || !retry_latest || !first_retry || retry->generation() != 1u ||
      retry_latest->generation() != 1u || !first_read ||
      observed != source_values || !first_overwrite) {
    return 7;
  }
  const Status failed = retry->run();
  const Stats failed_stats = retry->stats();
  if (failed || failed.reason() != Reason::GatherIndexOutOfRange ||
      retry->poisoned() || retry->generation() != 1u ||
      failed_stats.control.overflow_ordinal != 0u ||
      failed_stats.publication.commit_count != 1u ||
      failed_stats.publication.discard_count != 1u ||
      retry_latest->generation() != 1u ||
      !Overwrite(*indices, recovery_indices) || !retry->run() ||
      retry->generation() != 2u || !ReadExact(*retry, *source, observed) ||
      observed != recovered_values ||
      retry->stats().publication.commit_count != 2u ||
      retry->stats().publication.discard_count != 1u) {
    return 8;
  }

  constexpr std::array<std::int32_t, 4u> other_initial{10, 20, 30, 40};
  constexpr std::array<std::int32_t, 4u> other_once{11, 21, 31, 41};
  constexpr std::array<std::int32_t, 4u> other_twice{12, 22, 32, 42};
  auto left_first = Upload(device, initial);
  auto left_second = device.buffer<std::int32_t>(initial.size());
  auto right_first = Upload(device, other_initial);
  auto right_second = device.buffer<std::int32_t>(other_initial.size());
  if (!left_first || !left_second || !right_first || !right_second) {
    return 9;
  }
  auto paired = pipeline(device)
                    .state(*left_first, *left_second)
                    .state(*right_first, *right_second)
                    .then(*advance, read(*left_first), write(*left_second))
                    .then(*advance, read(*right_first), write(*right_second))
                    .commit()
                    .prepare();
  std::array<std::int32_t, initial.size()> left_observed{};
  std::array<std::int32_t, other_initial.size()> right_observed{};
  if (!paired || !paired->run() || paired->generation() != 1u ||
      !ReadExact(*paired, *left_second, left_observed) ||
      left_observed != once ||
      !ReadExact(*paired, *right_second, right_observed) ||
      right_observed != other_once || !paired->run() ||
      paired->generation() != 2u ||
      !ReadExact(*paired, *left_first, left_observed) ||
      left_observed != twice ||
      !ReadExact(*paired, *right_first, right_observed) ||
      right_observed != other_twice) {
    return 10;
  }
  const Stats before_paired_snapshot = paired->stats();
  const auto paired_snapshot = paired->snapshot();
  const Stats after_paired_snapshot = paired->stats();
  const std::uint64_t expected_download_events =
      backend == Backend::Cpu ? 0u : 2u;
  const std::uint64_t expected_downloaded_bytes =
      backend == Backend::Cpu ? 0u : sizeof(initial) + sizeof(other_initial);
  const std::uint64_t expected_snapshot_submits =
      backend == Backend::Vulkan ? 1u : 0u;
  if (!paired_snapshot ||
      after_paired_snapshot.publication.snapshot_byte_count !=
          sizeof(initial) + sizeof(other_initial) ||
      after_paired_snapshot.download_events !=
          before_paired_snapshot.download_events + expected_download_events ||
      after_paired_snapshot.downloaded_bytes !=
          before_paired_snapshot.downloaded_bytes + expected_downloaded_bytes ||
      after_paired_snapshot.command_submits !=
          before_paired_snapshot.command_submits + expected_snapshot_submits) {
    return 11;
  }

  auto poison_source = Upload(device, initial);
  auto poison_pending = device.buffer<std::int32_t>(initial.size());
  auto poison_indices = Upload(device, invalid_indices);
  auto poison_output = device.buffer<std::int32_t>(initial.size());
  if (!poison_source || !poison_pending || !poison_indices || !poison_output) {
    return 12;
  }
  auto mixed_failure =
      pipeline(device)
          .state(*poison_source, *poison_pending)
          .then(*advance, read(*poison_source), write(*poison_pending))
          .then(*gather, read(*poison_pending, *poison_indices),
                write(*poison_output))
          .commit()
          .prepare();
  const Status mixed_result = mixed_failure
                                  ? mixed_failure->run()
                                  : Status::fail(Reason::PipelineInvalid);
  const auto mixed_snapshot =
      mixed_failure ? mixed_failure->snapshot()
                    : Result<StateSnapshot>::fail(Reason::PipelineInvalid);
  const auto &published_state = detail::BufferAccess::state(*poison_source);
  const auto &pending_state = detail::BufferAccess::state(*poison_pending);
  const auto &ordinary_state = detail::BufferAccess::state(*poison_output);
  if (!mixed_failure || mixed_result ||
      mixed_result.reason() != Reason::GatherIndexOutOfRange ||
      !mixed_failure->poisoned() || mixed_failure->generation() != 0u ||
      !mixed_snapshot || mixed_snapshot->generation() != 0u ||
      mixed_failure->stats().publication.discard_count != 1u ||
      published_state == nullptr || pending_state == nullptr ||
      ordinary_state == nullptr || detail::buffer_poisoned(*published_state) ||
      detail::buffer_poisoned(*pending_state) ||
      !detail::buffer_poisoned(*ordinary_state)) {
    return 13;
  }

  auto mixed_first = Upload(device, initial);
  auto mixed_second = device.buffer<std::int32_t>(initial.size());
  auto mixed_output = device.buffer<std::int32_t>(initial.size());
  if (!mixed_first || !mixed_second || !mixed_output) {
    return 15;
  }
  auto mixed = pipeline(device)
                   .state(*mixed_first, *mixed_second)
                   .then(*advance, read(*mixed_first), write(*mixed_second))
                   .then(*advance, read(*mixed_second), write(*mixed_output))
                   .commit()
                   .prepare();
  std::array<std::int32_t, initial.size()> mixed_observed{};
  if (!mixed || !mixed->run() || mixed->stats().output_hash != 0u ||
      !ReadExact(*mixed, *mixed_output, mixed_observed) ||
      mixed_observed != twice || mixed->stats().output_hash != 0u ||
      !ReadExact(*mixed, *mixed_second, mixed_observed) ||
      mixed_observed != once || mixed->stats().output_hash == 0u) {
    return 16;
  }
  const std::uint64_t current_mixed_hash = mixed->stats().output_hash;
  if (mixed_hash == 0u) {
    mixed_hash = current_mixed_hash;
  } else if (mixed_hash != current_mixed_hash) {
    return 17;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
