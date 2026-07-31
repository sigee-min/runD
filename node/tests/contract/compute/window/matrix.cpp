#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace rund::node::test_contract::window {

[[nodiscard]] int CheckWindowMatrix(Device &device, const Backend backend,
                                    MatrixIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, kMatrixWidth> first_seed{10u, 20u, 30u,
                                                               40u};
  constexpr std::array<std::uint32_t, kMatrixWidth> second_seed{100u, 200u,
                                                                300u, 400u};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr auto invalid_count =
      PackedCount(static_cast<std::uint32_t>(kMatrixMaximum + 1u));
  constexpr std::array<std::uint32_t, 1u> safe_failure{
      static_cast<std::uint32_t>(kMatrixWindows)};
  constexpr std::array<std::uint32_t, kMatrixBacking> first_initial{
      901u, 902u, 903u, 904u, 905u, 906u, 907u, 908u, 909u};
  constexpr std::array<std::uint32_t, kMatrixBacking> second_initial{
      801u, 802u, 803u, 804u, 805u, 806u, 807u, 808u, 809u};
  constexpr std::size_t first_offset = 1u;
  constexpr std::size_t second_offset = 0u;
  constexpr std::size_t stride = 2u;
  constexpr std::array<std::uint32_t, 7u> boundaries{
      0u,
      1u,
      static_cast<std::uint32_t>(kMatrixTile - 1u),
      static_cast<std::uint32_t>(kMatrixTile),
      static_cast<std::uint32_t>(kMatrixTile + 1u),
      static_cast<std::uint32_t>(kMatrixMaximum - 1u),
      static_cast<std::uint32_t>(kMatrixMaximum)};

  auto body = MatrixBody(device);
  auto scalar_copy = on(device)
                         .map<std::uint32_t>("resident-window-control-copy", 1u,
                                             [](auto value) { return value; })
                         .compile();
  auto count_copy = on(device)
                        .map<std::uint32_t>("resident-window-count-copy",
                                            invalid_count.size(),
                                            [](auto value) { return value; })
                        .compile();
  auto backing_copy =
      on(device)
          .map<std::uint32_t>("resident-window-backing-copy", kMatrixBacking,
                              [](auto value) { return value; })
          .compile();
  auto first =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{first_seed});
  auto second =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{second_seed});
  auto witness = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{witness_values});
  auto fail_at = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{safe_failure});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{invalid_count});
  auto first_backing = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{first_initial});
  auto second_backing = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{second_initial});
  auto scratch = device.buffer<std::uint32_t>(kMatrixBacking);
  auto count_scratch = device.buffer<std::uint32_t>(invalid_count.size());
  if (!body || !scalar_copy || !count_copy || !backing_copy || !first ||
      !second || !witness || !fail_at || !count || !first_backing ||
      !second_backing || !scratch || !count_scratch) {
    return 1;
  }
  auto first_view = first_backing->view(first_offset, kMatrixWidth, stride);
  auto second_view = second_backing->view(second_offset, kMatrixWidth, stride);
  const auto &resident = *count;
  auto count_view = resident.view(1u, 1u);
  if (!first_view || !second_view || !count_view) {
    return 2;
  }

  auto builder = pipeline(device);
  builder.windows<kMatrixMaximum, kMatrixTile>(
      *body, rund::compute::window(*count_view),
      read(*first, *second, *witness, *fail_at),
      write(*first_view, *second_view));
  const auto plan = builder.plan();
  constexpr std::uint64_t published_bytes =
      2u * kMatrixWidth * sizeof(std::uint32_t);
  if (!plan || plan->publish_count != 2u ||
      plan->publish_bytes != published_bytes) {
    std::fprintf(
        stderr,
        "window matrix plan backend=%u status=%u reason=%u "
        "publishes=%llu bytes=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(plan.ok()),
        static_cast<unsigned>(plan.reason()),
        static_cast<unsigned long long>(plan ? plan->publish_count : 0u),
        static_cast<unsigned long long>(plan ? plan->publish_bytes : 0u));
    return 3;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan) {
    return 4;
  }
  const std::shared_ptr<rund::compute::detail::PipelineState> state =
      rund::compute::detail::PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->steps.size() != kMatrixWindows ||
      state->window_rank.size() != kMatrixWindows + 1u) {
    return 5;
  }
  for (std::size_t prefix = 0u; prefix <= kMatrixWindows; ++prefix) {
    if (state->window_rank[prefix] != prefix) {
      return 6;
    }
  }

  const Status invalid = prepared->run();
  const Stats invalid_stats = prepared->stats();
  const std::uint64_t submits = backend == Backend::Cpu ? 0u : 1u;
  std::array<std::uint32_t, kMatrixBacking> first_actual{};
  std::array<std::uint32_t, kMatrixBacking> second_actual{};
  std::array<std::uint32_t, 3u> count_actual{};
  const bool first_observed =
      Observe(*backing_copy, *first_backing, *scratch, first_actual);
  const bool second_observed =
      Observe(*backing_copy, *second_backing, *scratch, second_actual);
  const bool count_observed =
      Observe(*count_copy, *count, *count_scratch, count_actual);
  if (invalid || invalid.reason() != Reason::BoundedCountInvalid ||
      prepared->poisoned() || prepared->generation() != 0u ||
      invalid_stats.command_submits != submits ||
      invalid_stats.control.iteration_count != 0u ||
      invalid_stats.pipeline.verified_step_count != 0u ||
      invalid_stats.pipeline.failed_step_index != 0u || !first_observed ||
      !second_observed || !count_observed || first_actual != first_initial ||
      second_actual != second_initial || count_actual != invalid_count) {
    std::fprintf(
        stderr,
        "window matrix overflow backend=%u status=%u reason=%u poison=%u "
        "generation=%llu submits=%llu iterations=%llu verified=%llu "
        "failed=%llu observed=%u/%u/%u first=%u second=%u "
        "count=%08x/%08x/%08x\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(invalid.ok()),
        static_cast<unsigned>(invalid.reason()),
        static_cast<unsigned>(prepared->poisoned()),
        static_cast<unsigned long long>(prepared->generation()),
        static_cast<unsigned long long>(invalid_stats.command_submits),
        static_cast<unsigned long long>(invalid_stats.control.iteration_count),
        static_cast<unsigned long long>(
            invalid_stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(
            invalid_stats.pipeline.failed_step_index),
        static_cast<unsigned>(first_observed),
        static_cast<unsigned>(second_observed),
        static_cast<unsigned>(count_observed), first_actual[first_offset],
        second_actual[second_offset], count_actual[0u], count_actual[1u],
        count_actual[2u]);
    return 5;
  }

  std::uint64_t expected_generation = 0u;
  const auto check_success = [&](const std::uint32_t active,
                                 const int failure) -> int {
    if (!SetCount(device, *count_copy, *count, active)) {
      return failure;
    }
    const Status ran = prepared->run();
    const Stats stats = prepared->stats();
    ++expected_generation;
    const std::uint64_t active_windows =
        (static_cast<std::uint64_t>(active) + kMatrixTile - 1u) / kMatrixTile;
    const std::uint64_t skipped_windows = kMatrixWindows - active_windows;
    const std::uint64_t expected_iterations =
        backend == Backend::Cpu ? active_windows : kMatrixWindows;
    const std::uint64_t expected_skipped =
        backend == Backend::Cpu ? skipped_windows : 0u;
    const std::uint32_t sum = PrefixSum(active);
    std::array<std::uint32_t, kMatrixWidth> expected_first{};
    std::array<std::uint32_t, kMatrixWidth> expected_second{};
    for (std::size_t index = 0u; index < kMatrixWidth; ++index) {
      expected_first[index] = first_seed[index] + sum;
      expected_second[index] = second_seed[index] + sum + sum;
    }
    const auto expected_first_backing =
        Projected(first_initial, expected_first, first_offset, stride);
    const auto expected_second_backing =
        Projected(second_initial, expected_second, second_offset, stride);
    if (!ran || prepared->poisoned() ||
        prepared->generation() != expected_generation ||
        stats.command_submits != submits ||
        stats.control.iteration_count != expected_iterations ||
        stats.control.skipped_iteration_count != expected_skipped ||
        stats.pipeline.verified_step_count != 1u ||
        stats.pipeline.failed_step_index != PipelineStats::no_failed_step ||
        !prepared->read(*first_backing, first_actual) ||
        !prepared->read(*second_backing, second_actual) ||
        !Observe(*count_copy, *count, *count_scratch, count_actual) ||
        first_actual != expected_first_backing ||
        second_actual != expected_second_backing ||
        count_actual != PackedCount(active)) {
      std::fprintf(
          stderr,
          "window matrix success backend=%u count=%u status=%u reason=%u "
          "poison=%u generation=%llu expected_generation=%llu submits=%llu "
          "iterations=%llu verified=%llu failed=%llu first=%u/%u "
          "second=%u/%u count=%08x/%08x/%08x\n",
          static_cast<unsigned>(backend), active,
          static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
          static_cast<unsigned>(prepared->poisoned()),
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(expected_generation),
          static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(stats.control.iteration_count),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          first_actual[first_offset], expected_first[0],
          second_actual[second_offset], expected_second[0], count_actual[0u],
          count_actual[1u], count_actual[2u]);
      return failure;
    }
    return 0;
  };

  if (const int retry =
          check_success(static_cast<std::uint32_t>(kMatrixMaximum), 6);
      retry != 0) {
    return retry;
  }
  for (const std::uint32_t boundary : boundaries) {
    if (const int checked = check_success(boundary, 7); checked != 0) {
      return checked;
    }
  }

  const auto check_failure = [&](const std::uint32_t failure_at,
                                 const int failure) -> int {
    if (!SetCount(device, *count_copy, *count,
                  static_cast<std::uint32_t>(kMatrixMaximum)) ||
        !SetControl(device, *scalar_copy, *fail_at, failure_at)) {
      return failure;
    }
    const auto before_first = first_actual;
    const auto before_second = second_actual;
    const std::uint64_t before_generation = prepared->generation();
    const std::uint64_t before_discard =
        prepared->stats().publication.discard_count;
    const Status failed = prepared->run();
    const Stats stats = prepared->stats();
    if (failed || failed.reason() != Reason::GatherIndexOutOfRange ||
        prepared->poisoned() || prepared->generation() != before_generation ||
        stats.command_submits != submits ||
        stats.control.iteration_count != failure_at ||
        stats.pipeline.verified_step_count != 0u ||
        stats.pipeline.failed_step_index != 0u ||
        stats.publication.discard_count != before_discard + 1u ||
        !prepared->read(*first_backing, first_actual) ||
        !prepared->read(*second_backing, second_actual) ||
        !Observe(*count_copy, *count, *count_scratch, count_actual) ||
        first_actual != before_first || second_actual != before_second ||
        count_actual !=
            PackedCount(static_cast<std::uint32_t>(kMatrixMaximum))) {
      std::fprintf(
          stderr,
          "window matrix failure backend=%u at=%u status=%u reason=%u "
          "poison=%u generation=%llu/%llu submits=%llu iterations=%llu "
          "verified=%llu failed=%llu discard=%llu first=%u/%u "
          "second=%u/%u count=%08x/%08x/%08x\n",
          static_cast<unsigned>(backend), failure_at,
          static_cast<unsigned>(failed.ok()),
          static_cast<unsigned>(failed.reason()),
          static_cast<unsigned>(prepared->poisoned()),
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(before_generation),
          static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(stats.control.iteration_count),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(stats.publication.discard_count),
          first_actual[first_offset], before_first[first_offset],
          second_actual[second_offset], before_second[second_offset],
          count_actual[0u], count_actual[1u], count_actual[2u]);
      return failure;
    }
    return 0;
  };

  if (const int middle = check_failure(1u, 8); middle != 0) {
    return middle;
  }
  if (const int final = check_failure(2u, 9); final != 0) {
    return final;
  }
  if (!SetControl(device, *scalar_copy, *fail_at,
                  static_cast<std::uint32_t>(kMatrixWindows))) {
    return 10;
  }
  if (const int retry =
          check_success(static_cast<std::uint32_t>(kMatrixMaximum), 10);
      retry != 0) {
    return retry;
  }

  std::array<std::uint32_t, kMatrixBacking * 2u> raw{};
  std::copy(first_actual.begin(), first_actual.end(), raw.begin());
  std::copy(second_actual.begin(), second_actual.end(),
            raw.begin() + kMatrixBacking);
  const std::uint64_t output_hash = Hash(raw.data(), sizeof(raw));
  const Fingerprint body_hash = body->fingerprint();
  const Fingerprint pipeline_hash = prepared->fingerprint();
  if (!body_hash || !pipeline_hash || output_hash == 0u) {
    return 11;
  }
  if (identity.body) {
    if (identity.body != body_hash || identity.pipeline != pipeline_hash ||
        identity.output != output_hash) {
      std::fprintf(
          stderr,
          "window matrix identity backend=%u body=%016llx:%016llx/"
          "%016llx:%016llx pipeline=%016llx:%016llx/%016llx:%016llx "
          "output=%016llx/%016llx\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(body_hash.hi),
          static_cast<unsigned long long>(body_hash.lo),
          static_cast<unsigned long long>(identity.body.hi),
          static_cast<unsigned long long>(identity.body.lo),
          static_cast<unsigned long long>(pipeline_hash.hi),
          static_cast<unsigned long long>(pipeline_hash.lo),
          static_cast<unsigned long long>(identity.pipeline.hi),
          static_cast<unsigned long long>(identity.pipeline.lo),
          static_cast<unsigned long long>(output_hash),
          static_cast<unsigned long long>(identity.output));
      return 12;
    }
  } else {
    identity.body = body_hash;
    identity.pipeline = pipeline_hash;
    identity.output = output_hash;
  }
  return 0;
}

} // namespace rund::node::test_contract::window
