#include "local.hpp"

#include "../../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>

namespace runtime_compute_pipeline_accel_detail {
namespace {

int CheckObservation(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input{3, 2, 1, 0};
  auto program =
      on(device)
          .input<std::int32_t>(input.size())
          .branch([](auto values) {
            auto active = values.filter(
                [](auto value) { return value > std::int32_t{-100}; });
            return active.template unroll<2u>(
                [](auto work) {
                  return work.map("observe-step", [](auto value) {
                    return value - std::int32_t{1};
                  });
                },
                [](auto value) { return value == std::int32_t{99}; });
          })
          .compile();
  auto source = device.upload<std::int32_t>(input);
  auto output = device.buffer<std::int32_t>(input.size());
  auto count = device.buffer<std::uint32_t>(1u);
  if (!program || !source || !output || !count) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .then(*program, read(*source), write(*output, *count))
                      .prepare();
  if (!prepared) {
    std::fprintf(stderr, "pipeline observation prepare failed: %.*s\n",
                 static_cast<int>(prepared.error().size()),
                 prepared.error().data());
    return 2;
  }
  const Status status = prepared->run();
  if (!status) {
    std::fprintf(stderr, "pipeline observation run failed: %.*s\n",
                 static_cast<int>(status.error().size()),
                 status.error().data());
    return 2;
  }
  const Stats stats = prepared->stats();
  std::array<std::int32_t, input.size()> values{};
  std::array<std::uint32_t, 1u> size{};
  if (stats.control.iteration_count != 2u ||
      stats.control.skipped_iteration_count != 0u ||
      !prepared->read(*output, values) || !prepared->read(*count, size) ||
      values != std::array<std::int32_t, input.size()>{1, 0, -1, -2} ||
      size[0u] != input.size()) {
    std::fprintf(
        stderr, "pipeline observation iterations=%llu/%llu count=%u\n",
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.control.skipped_iteration_count),
        size[0u]);
    return 3;
  }
  return 0;
}

int CheckWindow(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 8u;
  constexpr std::size_t tile = 3u;
  constexpr std::array<std::uint32_t, 1u> state_seed{7u};
  constexpr std::array<std::uint32_t, 1u> terminal_seed{0u};
  constexpr std::array<std::uint32_t, 1u> count_seed{maximum};
  auto body =
      on(device)
          .input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .branch([](auto state, auto terminal, auto count, auto ordinal) {
            (void)count;
            (void)ordinal;
            auto next = state.map("session-window-state",
                                  [](auto value) { return value + 1u; });
            auto stop = terminal.map("session-window-terminal", [](auto value) {
              return value * 0u + 7u;
            });
            return outputs(next, stop);
          })
          .compile();
  auto initial = device.upload<std::uint32_t>(state_seed);
  auto terminal = device.upload<std::uint32_t>(terminal_seed);
  auto count = device.upload<std::uint32_t>(count_seed);
  auto output = device.buffer<std::uint32_t>(1u);
  auto stopped = device.buffer<std::uint32_t>(1u);
  if (!body || !initial || !terminal || !count || !output || !stopped) {
    return 1;
  }
  auto prepared =
      pipeline(device)
          .windows<maximum, tile>(
              *body, rund::compute::window(*count).until<1u>(7u),
              read(*initial, *terminal), write_final(*output, *stopped))
          .prepare();
  if (!prepared) {
    return 2;
  }
  const Completion completion = session.compute(*prepared).submit().wait();
  std::array<std::uint32_t, 1u> actual{};
  std::array<std::uint32_t, 1u> terminal_actual{};
  if (!completion || completion.stats().command_submits != 1u ||
      completion.stats().pipeline.step_count != 1u ||
      completion.stats().pipeline.verified_step_count != 1u ||
      completion.stats().control.iteration_count != 3u ||
      !prepared->read(*output, actual) ||
      !prepared->read(*stopped, terminal_actual) || actual[0] != 8u ||
      terminal_actual[0] != 7u) {
    return 3;
  }
  return 0;
}

} // namespace

int CheckPipelineControl(rund::Session &session,
                         rund::compute::Device &device) {
  if (const int observed = CheckObservation(device); observed != 0) {
    return 70 + observed;
  }
  if (const int window = CheckWindow(session, device); window != 0) {
    return 75 + window;
  }
  return 0;
}

} // namespace runtime_compute_pipeline_accel_detail
