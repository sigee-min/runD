#include "local.hpp"

#include <cstdio>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSealedRepetitions(
    rund::compute::Device &device, const Backend backend,
    rund::compute::graph::Fingerprint &canonical_fingerprint) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> values{1, 3, 5, 7};
  auto transform =
      on(device)
          .map<std::int32_t>("pipeline-sealed-repetition-transform",
                             values.size(),
                             [](auto value) { return value * 3 + 1; })
          .compile();
  auto input = Upload(device, values);
  auto canonical_output = device.buffer<std::int32_t>(values.size());
  auto sealed_output = device.buffer<std::int32_t>(values.size());
  auto equivalent_input = Upload(device, values);
  auto equivalent_output = device.buffer<std::int32_t>(values.size());
  if (!transform || !input || !canonical_output || !sealed_output ||
      !equivalent_input || !equivalent_output) {
    return 1;
  }

  auto canonical = pipeline(device)
                       .then(*transform, read(*input), write(*canonical_output))
                       .prepare();
  auto unit_sealed =
      pipeline(device)
          .sealed_repetitions<1u>()
          .then(*transform, read(*equivalent_input), write(*equivalent_output))
          .prepare();
  auto sealed = pipeline(device)
                    .sealed_repetitions<256u>()
                    .then(*transform, read(*input), write(*sealed_output))
                    .prepare();
  auto equivalent =
      pipeline(device)
          .sealed_repetitions<256u>()
          .then(*transform, read(*equivalent_input), write(*equivalent_output))
          .prepare();
  auto sealed_two =
      pipeline(device)
          .sealed_repetitions<2u>()
          .then(*transform, read(*equivalent_input), write(*equivalent_output))
          .prepare();
  auto sealed_four =
      pipeline(device)
          .sealed_repetitions<4u>()
          .then(*transform, read(*equivalent_input), write(*equivalent_output))
          .prepare();
  if (!canonical || !unit_sealed || !sealed || !equivalent || !sealed_two ||
      !sealed_four || unit_sealed->fingerprint() != canonical->fingerprint() ||
      sealed->fingerprint() == canonical->fingerprint() ||
      equivalent->fingerprint() != sealed->fingerprint() ||
      sealed_two->fingerprint() == sealed_four->fingerprint()) {
    return 2;
  }
  if (canonical_fingerprint) {
    if (canonical_fingerprint != sealed->fingerprint()) {
      return 3;
    }
  } else {
    canonical_fingerprint = sealed->fingerprint();
  }

  if (sealed->stats().pipeline.sealed_repetition_count != 256u ||
      sealed->stats().pipeline.coalesced_repetition_count != 0u ||
      !canonical->run() || !sealed->run()) {
    return 4;
  }
  const Stats canonical_stats = canonical->stats();
  const Stats sealed_stats = sealed->stats();
  std::array<std::int32_t, values.size()> canonical_values{};
  std::array<std::int32_t, values.size()> sealed_values{};
  constexpr std::array<std::int32_t, values.size()> expected{4, 10, 16, 22};
  if (!ReadExact(*canonical, *canonical_output, canonical_values) ||
      !ReadExact(*sealed, *sealed_output, sealed_values) ||
      canonical_values != expected || sealed_values != canonical_values ||
      sealed->generation() != 1u ||
      sealed_stats.pipeline.sealed_repetition_count != 256u ||
      sealed_stats.pipeline.coalesced_repetition_count != 255u ||
      sealed_stats.pipeline.step_count != 1u ||
      sealed_stats.pipeline.verified_step_count != 1u ||
      sealed_stats.pipeline.failed_step_index !=
          PipelineStats::no_failed_step ||
      sealed_stats.dispatches != canonical_stats.dispatches ||
      sealed_stats.command_submits != canonical_stats.command_submits ||
      sealed_stats.pipeline.prepared_command_count !=
          canonical_stats.pipeline.prepared_command_count) {
    std::fprintf(
        stderr,
        "pipeline sealed repetitions backend=%u repetitions=%llu "
        "coalesced=%llu "
        "dispatches=%llu/%llu submits=%llu/%llu generation=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(
            sealed_stats.pipeline.sealed_repetition_count),
        static_cast<unsigned long long>(
            sealed_stats.pipeline.coalesced_repetition_count),
        static_cast<unsigned long long>(canonical_stats.dispatches),
        static_cast<unsigned long long>(sealed_stats.dispatches),
        static_cast<unsigned long long>(canonical_stats.command_submits),
        static_cast<unsigned long long>(sealed_stats.command_submits),
        static_cast<unsigned long long>(sealed->generation()));
    return 5;
  }
  if (!sealed->run() || sealed->generation() != 2u ||
      sealed->stats().pipeline.coalesced_repetition_count != 255u ||
      !ReadExact(*sealed, *sealed_output, sealed_values) ||
      sealed_values != expected) {
    return 6;
  }

  auto shared =
      Upload(device, std::array<std::int32_t, 8u>{1, 2, 3, 4, 5, 6, 7, 8});
  auto strided = on(device)
                     .map<std::int32_t>("pipeline-sealed-strided", 4u,
                                        [](auto value) { return value + 10; })
                     .compile();
  if (!shared || !strided) {
    return 7;
  }
  auto even = shared->view(0u, 4u, 2u);
  auto odd = shared->view(1u, 4u, 2u);
  if (!even || !odd) {
    return 8;
  }
  auto disjoint = pipeline(device)
                      .sealed_repetitions<8u>()
                      .then(*strided, read(*even), write(*odd))
                      .prepare();
  std::array<std::int32_t, 8u> strided_values{};
  if (!disjoint || !disjoint->run() ||
      !ReadExact(*disjoint, *shared, strided_values) ||
      strided_values !=
          std::array<std::int32_t, 8u>{1, 11, 3, 13, 5, 15, 7, 17} ||
      disjoint->stats().pipeline.coalesced_repetition_count != 7u) {
    return 9;
  }

  auto overlap_transform =
      on(device)
          .map<std::int32_t>("pipeline-sealed-overlap", 2u,
                             [](auto value) { return value + 20; })
          .compile();
  auto overlap_read = shared->view(0u, 2u, 2u);
  auto overlap_write = shared->view(2u, 2u, 2u);
  if (!overlap_transform || !overlap_read || !overlap_write) {
    return 10;
  }
  auto overlap_builder = pipeline(device).sealed_repetitions<8u>().then(
      *overlap_transform, read(*overlap_read), write(*overlap_write));
  const auto overlap_plan = overlap_builder.plan();
  auto overlap = std::move(overlap_builder).prepare();
  if (overlap_plan ||
      overlap_plan.reason() != Reason::PipelineTemporalDependency || overlap ||
      overlap.reason() != Reason::PipelineTemporalDependency) {
    return 11;
  }

  auto feedback_a = Upload(device, values);
  auto feedback_b = device.buffer<std::int32_t>(values.size());
  if (!feedback_a || !feedback_b) {
    return 12;
  }
  auto feedback_builder =
      pipeline(device)
          .sealed_repetitions<8u>()
          .then(*transform, read(*feedback_a), write(*feedback_b))
          .then(*transform, read(*feedback_b), write(*feedback_a));
  const auto feedback_plan = feedback_builder.plan();
  auto feedback = std::move(feedback_builder).prepare();
  if (feedback_plan ||
      feedback_plan.reason() != Reason::PipelineTemporalDependency ||
      feedback || feedback.reason() != Reason::PipelineTemporalDependency) {
    return 13;
  }

  auto window_body =
      on(device)
          .input<std::int32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .branch([](auto state, auto count, auto ordinal) {
            (void)count;
            (void)ordinal;
            return state.map("pipeline-sealed-window-publication",
                             [](auto value) { return value + 1; });
          })
          .compile();
  constexpr std::array<std::uint32_t, 1u> window_count_value{4u};
  constexpr std::array<std::int32_t, 1u> window_state_value{1};
  auto window_count = Upload(device, window_count_value);
  auto window_state = Upload(device, window_state_value);
  if (!window_body || !window_count || !window_state) {
    return 14;
  }
  auto publication_builder = pipeline(device).sealed_repetitions<2u>();
  publication_builder.windows<4u, 1u>(*window_body, window(*window_count),
                                      read(*window_state),
                                      write_final(*window_state));
  const auto publication_plan = publication_builder.plan();
  auto publication = std::move(publication_builder).prepare();
  if (publication_plan ||
      publication_plan.reason() != Reason::PipelineTemporalDependency ||
      publication ||
      publication.reason() != Reason::PipelineTemporalDependency) {
    return 15;
  }

  auto published = Upload(device, values);
  auto pending = device.buffer<std::int32_t>(values.size());
  if (!published || !pending) {
    return 16;
  }
  auto transactional_builder =
      pipeline(device)
          .sealed_repetitions<8u>()
          .state(*published, *pending)
          .then(*transform, read(*published), write(*pending))
          .commit();
  const auto transactional_plan = transactional_builder.plan();
  auto transactional = std::move(transactional_builder).prepare();
  if (transactional_plan ||
      transactional_plan.reason() != Reason::PipelineTemporalDependency ||
      transactional ||
      transactional.reason() != Reason::PipelineTemporalDependency) {
    return 17;
  }

  auto ordinary_published = Upload(device, values);
  auto ordinary_pending = device.buffer<std::int32_t>(values.size());
  auto unit_published = Upload(device, values);
  auto unit_pending = device.buffer<std::int32_t>(values.size());
  if (!ordinary_published || !ordinary_pending || !unit_published ||
      !unit_pending) {
    return 22;
  }
  auto ordinary_transaction =
      pipeline(device)
          .state(*ordinary_published, *ordinary_pending)
          .then(*transform, read(*ordinary_published), write(*ordinary_pending))
          .commit()
          .prepare();
  auto unit_transaction =
      pipeline(device)
          .sealed_repetitions<1u>()
          .state(*unit_published, *unit_pending)
          .then(*transform, read(*unit_published), write(*unit_pending))
          .commit()
          .prepare();
  std::array<std::int32_t, values.size()> ordinary_transaction_values{};
  std::array<std::int32_t, values.size()> unit_transaction_values{};
  if (!ordinary_transaction || !unit_transaction ||
      ordinary_transaction->fingerprint() != unit_transaction->fingerprint() ||
      !ordinary_transaction->run() || !unit_transaction->run() ||
      ordinary_transaction->generation() != 1u ||
      unit_transaction->generation() != 1u ||
      unit_transaction->stats().pipeline.sealed_repetition_count != 1u ||
      unit_transaction->stats().pipeline.coalesced_repetition_count != 0u ||
      !ReadExact(*ordinary_transaction, *ordinary_pending,
                 ordinary_transaction_values) ||
      !ReadExact(*unit_transaction, *unit_pending, unit_transaction_values) ||
      ordinary_transaction_values != expected ||
      unit_transaction_values != ordinary_transaction_values) {
    return 23;
  }

  constexpr std::array<std::int64_t, 4u> gather_values{1, 2, 3, 4};
  constexpr std::array<std::uint32_t, 1u> invalid_index{4u};
  auto gather = on(device)
                    .input<std::int64_t>(gather_values.size())
                    .zip_input<std::uint32_t>(invalid_index.size())
                    .branch([](auto source, auto indices) {
                      return source.gather(indices);
                    })
                    .compile();
  auto gather_input = Upload(device, gather_values);
  auto gather_index = Upload(device, invalid_index);
  auto gather_output = device.buffer<std::int64_t>(invalid_index.size());
  if (!gather || !gather_input || !gather_index || !gather_output) {
    return 18;
  }
  auto failing = pipeline(device)
                     .sealed_repetitions<8u>()
                     .then(*gather, read(*gather_input, *gather_index),
                           write(*gather_output))
                     .prepare();
  if (!failing) {
    return 19;
  }
  const Status failed = failing->run();
  if (failed || failed.reason() != Reason::GatherIndexOutOfRange ||
      failing->generation() != 0u || !failing->poisoned() ||
      failing->stats().publication.generation != 0u ||
      failing->stats().pipeline.sealed_repetition_count != 8u ||
      failing->stats().pipeline.coalesced_repetition_count != 0u) {
    return 20;
  }

  auto duplicated = pipeline(device)
                        .sealed_repetitions<2u>()
                        .sealed_repetitions<4u>()
                        .then(*transform, read(*input), write(*sealed_output))
                        .prepare();
  return !duplicated && duplicated.reason() == Reason::PipelineInvalid ? 0 : 21;
}

} // namespace rund_node_test_pipeline
