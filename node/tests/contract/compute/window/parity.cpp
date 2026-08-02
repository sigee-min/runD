#include "../pipeline/local.hpp"
#include "local.hpp"

#include <cstdio>
#include <utility>

namespace rund::node::test_contract::window {
namespace {

template <class T>
[[nodiscard]] int
CheckParity(Device &device, const Backend backend, Fingerprint &body_identity,
            Fingerprint &pipeline_identity, std::uint64_t &output_identity) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 10u;
  constexpr std::size_t tile = 4u;
  constexpr std::size_t active = 9u;
  constexpr std::size_t windows = CeilDiv(maximum, tile);
  static_assert(windows == 3u);
  static_assert(ResidentWindow<maximum, tile>::window_count == windows);

  const auto input_values = Values<T, maximum>();
  constexpr typename T::Raw seed_raw = 5;
  constexpr std::array<T, 1u> seed{T::from_raw(seed_raw)};
  const T sentinel = T::from_raw(static_cast<typename T::Raw>(-97));
  const auto output_values = Filled<T, 1u>(sentinel);
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(active)};

  auto body = Fold<T, maximum, tile>(device);
  auto initial = device.upload<T>(std::span<const T>{seed});
  auto input = device.upload<T>(std::span<const T>{input_values});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.upload<T>(std::span<const T>{output_values});
  if (!body || !initial || !input || !count || !output) {
    return 1;
  }

  auto builder = pipeline(device);
  builder.profile(PipelineProfile::Steps);
  builder.template windows<maximum, tile>(*body, rund::compute::window(*count),
                                          read(*initial, *input),
                                          write_final(*output));
  const auto planned = builder.plan();
  if (!planned ||
      planned->persistent_bytes <
          (maximum + 2u) * sizeof(T) + sizeof(std::uint32_t) ||
      planned->state_bytes < sizeof(T) || planned->transient_bytes == 0u ||
      planned->peak_bytes < planned->transient_bytes ||
      planned->peak_iteration >= windows || planned->peak_step != 0u) {
    if (planned) {
      std::fprintf(
          stderr,
          "window plan backend=%u persistent=%llu state=%llu transient=%llu "
          "peak=%llu iteration=%zu step=%zu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(planned->persistent_bytes),
          static_cast<unsigned long long>(planned->state_bytes),
          static_cast<unsigned long long>(planned->transient_bytes),
          static_cast<unsigned long long>(planned->peak_bytes),
          planned->peak_iteration, planned->peak_step);
    }
    return 2;
  }

  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = planned->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *planned) {
    std::fprintf(stderr, "window prepare backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(prepared.reason()));
    return 3;
  }
  if (!prepared->run() || !prepared->run()) {
    return 4;
  }
  std::array<PipelineStepProfile, windows> rows{};
  const auto profile = prepared->profile(rows);
  const std::uint64_t planned_resident = planned->state_bytes +
                                         planned->transient_bytes +
                                         planned->prepared_buffer_bytes;
  if (!profile ||
      profile->referenced_resource_bytes != planned->persistent_bytes ||
      profile->shared_memory.resident.current != planned_resident ||
      !rund_node_test_pipeline::ProfileMemoryReconciles(*profile, rows)) {
    return 10;
  }
  const auto stats = prepared->stats();
  const std::uint64_t submits = backend == Backend::Cpu ? 0u : 1u;
  const bool stats_match =
      stats.command_submits == submits && stats.pipeline.step_count == 1u &&
      stats.pipeline.verified_step_count == 1u &&
      stats.pipeline.failed_step_index ==
          rund::compute::PipelineStats::no_failed_step &&
      stats.control.iteration_count == windows &&
      stats.control.skipped_iteration_count == 0u &&
      stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
      stats.uploaded_bytes == 0u && stats.download_events == 0u;
  if (!stats_match) {
    std::fprintf(
        stderr,
        "window stats backend=%u submits=%llu steps=%llu verified=%llu "
        "failed=%llu iterations=%llu skipped=%llu compiles=%llu "
        "allocations=%llu uploaded=%llu downloads=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.control.skipped_iteration_count),
        static_cast<unsigned long long>(stats.pipeline_compiles),
        static_cast<unsigned long long>(stats.buffer_allocations),
        static_cast<unsigned long long>(stats.uploaded_bytes),
        static_cast<unsigned long long>(stats.download_events));
  }

  std::array<T, 1u> actual{};
  if (!prepared->read(*output, actual)) {
    return 6;
  }
  typename T::Raw expected_raw = seed_raw;
  for (std::size_t index = 0u; index < active; ++index) {
    expected_raw = static_cast<typename T::Raw>(
        expected_raw + static_cast<typename T::Raw>(index + 1u));
  }
  if (actual[0] != T::from_raw(expected_raw)) {
    return 7;
  }

  const std::uint64_t hash = Hash(actual.data(), sizeof(actual));
  const Fingerprint current_body = body->fingerprint();
  const Fingerprint current_pipeline = prepared->fingerprint();
  if (!current_body || !current_pipeline || hash == 0u) {
    return 8;
  }
  if (body_identity) {
    if (body_identity != current_body ||
        pipeline_identity != current_pipeline || output_identity != hash) {
      return 9;
    }
  } else {
    body_identity = current_body;
    pipeline_identity = current_pipeline;
    output_identity = hash;
  }
  return stats_match ? 0 : 5;
}

template <class T>
[[nodiscard]] int CheckOverflow(Device &device, const Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 10u;
  constexpr std::size_t tile = 4u;
  const auto input_values = Values<T, maximum>();
  constexpr std::array<T, 1u> seed{T::from_raw(7)};
  const T sentinel = T::from_raw(static_cast<typename T::Raw>(-113));
  const auto output_values = Filled<T, 1u>(sentinel);
  constexpr std::array<std::uint32_t, 1u> invalid_count{
      static_cast<std::uint32_t>(maximum + 1u)};

  auto body = Fold<T, maximum, tile>(device);
  auto observer = on(device)
                      .template map<T>("resident-window-observe", 1u,
                                       [](auto value) { return value; })
                      .compile();
  auto initial = device.upload<T>(std::span<const T>{seed});
  auto input = device.upload<T>(std::span<const T>{input_values});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{invalid_count});
  auto output = device.upload<T>(std::span<const T>{output_values});
  auto observed = device.buffer<T>(1u);
  if (!body || !observer || !initial || !input || !count || !output ||
      !observed) {
    return 1;
  }

  auto builder = pipeline(device);
  builder.template windows<maximum, tile>(*body, rund::compute::window(*count),
                                          read(*initial, *input),
                                          write_final(*output));
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan) {
    return 3;
  }
  const auto failed = prepared->run();
  const auto stats = prepared->stats();
  const std::uint64_t submits = backend == Backend::Cpu ? 0u : 1u;
  const bool overflow_match =
      !failed && failed.reason() == Reason::BoundedCountInvalid &&
      prepared->generation() == 0u && stats.command_submits == submits &&
      stats.pipeline.verified_step_count == 0u &&
      stats.pipeline.failed_step_index == 0u &&
      stats.control.overflow_ordinal == maximum;
  if (!overflow_match) {
    std::fprintf(
        stderr,
        "window overflow backend=%u status=%u reason=%u generation=%llu "
        "submits=%llu verified=%llu failed=%llu ordinal=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(failed.ok()),
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.control.overflow_ordinal));
    return 4;
  }

  auto copied = observer->run(*output, *observed);
  std::array<T, 1u> actual{};
  if (!copied) {
    std::fprintf(stderr, "window atomic copy backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(copied.reason()));
    return 5;
  }
  const auto readback = copied->read(*observed, std::span<T>{actual});
  if (!readback) {
    std::fprintf(stderr, "window atomic read backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(readback.reason()));
    return 5;
  }
  if (actual != output_values) {
    std::fprintf(
        stderr, "window atomic payload backend=%u actual=%lld expected=%lld\n",
        static_cast<unsigned>(backend), static_cast<long long>(actual[0].raw()),
        static_cast<long long>(sentinel.raw()));
    return 5;
  }
  return overflow_match ? 0 : 4;
}

} // namespace

[[nodiscard]] int CheckParity32(Device &device, const Backend backend,
                                Identity &identity) {
  return CheckParity<Fixed<16, 16>>(device, backend, identity.body32,
                                    identity.pipeline32, identity.output32);
}

[[nodiscard]] int CheckParity64(Device &device, const Backend backend,
                                Identity &identity) {
  return CheckParity<Fixed<20, 44>>(device, backend, identity.body64,
                                    identity.pipeline64, identity.output64);
}

[[nodiscard]] int CheckOverflow32(Device &device, const Backend backend) {
  return CheckOverflow<Fixed<16, 16>>(device, backend);
}

[[nodiscard]] int CheckOverflow64(Device &device, const Backend backend) {
  return CheckOverflow<Fixed<20, 44>>(device, backend);
}

} // namespace rund::node::test_contract::window
