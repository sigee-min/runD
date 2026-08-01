#include "local.hpp"

#include <array>
#include <cstdio>
#include <utility>

namespace rund_node_test_pipeline {
namespace {

template <std::size_t Iterations, std::size_t Count>
[[nodiscard]] constexpr std::array<std::int32_t, Iterations * Count>
ExpectedHistory(const std::array<std::int32_t, Count> &seed) noexcept {
  std::array<std::int32_t, Iterations * Count> expected{};
  for (std::size_t iteration = 0u; iteration < Iterations; ++iteration) {
    for (std::size_t element = 0u; element < Count; ++element) {
      expected[iteration * Count + element] =
          seed[element] + static_cast<std::int32_t>(iteration + 1u);
    }
  }
  return expected;
}

} // namespace

[[nodiscard]] int CheckIterationHistory(rund::compute::Device &device,
                                        const Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t iterations = 4u;
  constexpr std::size_t count = 4u;
  constexpr std::array<std::int32_t, count> seed{1, 3, 5, 7};
  constexpr auto expected = ExpectedHistory<iterations>(seed);

  auto input = Upload(device, seed);
  auto history = device.buffer<std::int32_t>(iterations * count);
  auto terminal = device.buffer<std::int32_t>(count);
  auto malformed = device.buffer<std::int32_t>(iterations * count - 1u);
  auto wrong_slice = device.buffer<std::int32_t>(iterations * (count + 1u));
  auto body = on(device)
                  .map<std::int32_t>("pipeline iteration history", count,
                                     [](auto value) { return value + 1; })
                  .compile();
  if (!input || !history || !terminal || !malformed || !wrong_slice || !body) {
    return 1;
  }

  auto bad =
      pipeline(device)
          .repeat<iterations>(*body, read(*input), write_each(*malformed))
          .prepare();
  if (bad || bad.reason() != Reason::ShapeMismatch) {
    return 2;
  }
  auto bad_slice =
      pipeline(device)
          .repeat<iterations>(*body, read(*input), write_each(*wrong_slice))
          .prepare();
  if (bad_slice || bad_slice.reason() != Reason::ShapeMismatch) {
    return 2;
  }

  auto history_builder =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .repeat<iterations>(*body, read(*input), write_each(*history));
  auto terminal_builder =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .repeat<iterations>(*body, read(*input), write_final(*terminal));
  const auto history_plan = history_builder.plan();
  const auto terminal_plan = terminal_builder.plan();
  constexpr std::uint64_t carried_bytes = count * sizeof(std::int32_t);
  if (!history_plan || !terminal_plan ||
      terminal_plan->state_bytes != history_plan->state_bytes + carried_bytes ||
      terminal_plan->allocation_count != history_plan->allocation_count + 1u) {
    std::fprintf(stderr,
                 "history plan backend=%u history_state=%llu "
                 "terminal_state=%llu history_alloc=%llu terminal_alloc=%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(
                     history_plan ? history_plan->state_bytes : 0u),
                 static_cast<unsigned long long>(
                     terminal_plan ? terminal_plan->state_bytes : 0u),
                 static_cast<unsigned long long>(
                     history_plan ? history_plan->allocation_count : 0u),
                 static_cast<unsigned long long>(
                     terminal_plan ? terminal_plan->allocation_count : 0u));
    return 3;
  }
  auto history_pipeline = std::move(history_builder).prepare();
  auto terminal_pipeline = std::move(terminal_builder).prepare();
  if (!history_pipeline || !terminal_pipeline ||
      history_pipeline->fingerprint() == terminal_pipeline->fingerprint()) {
    return 3;
  }

  const Status first = history_pipeline->run();
  const Stats first_stats = history_pipeline->stats();
  std::array<std::int32_t, iterations * count> observed{};
  if (!first || history_pipeline->generation() != 1u ||
      !ReadExact(*history_pipeline, *history, observed) ||
      observed != expected ||
      (backend != Backend::Cpu &&
       (first_stats.command_submits != 1u || first_stats.dispatches != 1u))) {
    std::fprintf(
        stderr,
        "history first backend=%u reason=%u generation=%llu "
        "submits=%llu dispatches=%llu exact=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(first.reason()),
        static_cast<unsigned long long>(history_pipeline->generation()),
        static_cast<unsigned long long>(first_stats.command_submits),
        static_cast<unsigned long long>(first_stats.dispatches),
        observed == expected ? 1u : 0u);
    return 4;
  }

  if (!history_pipeline->run()) {
    return 5;
  }
  const Stats warm = history_pipeline->stats();
  if (!WarmCountersClean(warm) ||
      warm.command_submits != first_stats.command_submits ||
      warm.dispatches != first_stats.dispatches ||
      history_pipeline->generation() != 2u) {
    return 5;
  }

  auto unit_final_buffer = device.buffer<std::int32_t>(count);
  auto unit_each_buffer = device.buffer<std::int32_t>(count);
  auto unit_final = unit_final_buffer
                        ? pipeline(device)
                              .repeat<1u>(*body, read(*input),
                                          write_final(*unit_final_buffer))
                              .prepare()
                        : Result<Pipeline>::fail(Reason::PipelineInvalid);
  auto unit_each =
      unit_each_buffer
          ? pipeline(device)
                .repeat<1u>(*body, read(*input), write_each(*unit_each_buffer))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!unit_final || !unit_each ||
      unit_final->fingerprint() != unit_each->fingerprint()) {
    return 6;
  }

  constexpr std::int32_t sentinel = -91;
  std::array<std::int32_t, iterations * count * 2u> strided_initial{};
  strided_initial.fill(sentinel);
  auto strided_history = Upload(device, strided_initial);
  auto strided_view =
      strided_history
          ? strided_history->view(1u, iterations * count, 2u)
          : Result<View<std::int32_t>>::fail(Reason::ResourceAccessInvalid);
  auto strided = strided_view
                     ? pipeline(device)
                           .repeat<iterations>(*body, read(*input),
                                               write_each(*strided_view))
                           .prepare()
                     : Result<Pipeline>::fail(Reason::PipelineInvalid);
  std::array<std::int32_t, iterations * count * 2u> strided_observed{};
  if (!strided || !strided->run() ||
      !ReadExact(*strided, *strided_history, strided_observed)) {
    return 7;
  }
  for (std::size_t index = 0u; index < strided_observed.size(); ++index) {
    const std::int32_t wanted =
        (index & 1u) == 0u ? sentinel : expected[index / 2u];
    if (strided_observed[index] != wanted) {
      return 8;
    }
  }
  if (backend != Backend::Cpu && (strided->stats().dispatches != 1u ||
                                  strided->stats().command_submits !=
                                      (backend == Backend::Vulkan ? 2u : 1u))) {
    return 9;
  }

  constexpr std::array<std::int32_t, count> failing_seed{2, 2, 2, 2};
  std::array<std::int32_t, iterations * count> failing_initial{};
  failing_initial.fill(sentinel);
  auto failing_input = Upload(device, failing_seed);
  auto failing_history = Upload(device, failing_initial);
  auto failing_body = on(device)
                          .map<std::int32_t>("pipeline history failure", count,
                                             [](auto value) {
                                               return (value - 1) / (value - 1);
                                             })
                          .compile();
  auto failing =
      failing_input && failing_history && failing_body
          ? pipeline(device)
                .repeat<iterations>(*failing_body, read(*failing_input),
                                    write_each(*failing_history))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!failing) {
    return 10;
  }
  const Status failed = failing->run();
  if (backend == Backend::Cpu) {
    std::array<std::int32_t, iterations * count> unreadable{};
    const Status rejected =
        failing->read(*failing_history, std::span<std::int32_t>{unreadable});
    if (failed.reason() != Reason::IntegerDivideByZero ||
        !failing->poisoned() || failing->generation() != 0u ||
        rejected.reason() != Reason::BufferPoisoned) {
      return 11;
    }
  } else if (failing->stats().dispatches != iterations) {
    std::fprintf(
        stderr,
        "history non-total backend=%u run=%u dispatches=%llu "
        "submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(failed.reason()),
        static_cast<unsigned long long>(failing->stats().dispatches),
        static_cast<unsigned long long>(failing->stats().command_submits));
    return 11;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
