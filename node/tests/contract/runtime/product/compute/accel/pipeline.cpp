#include "../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>

namespace {

template <class T>
int CheckFixedRecurrence(rund::compute::Device &device,
                         const std::array<T, 4u> &seed, const T factor,
                         const char *const name) {
  using namespace rund::compute;
  constexpr std::size_t iterations = 17u;
  auto body = on(device)
                  .template map<T>(name, seed.size(),
                                   capture(
                                       [](auto value, auto scale) {
                                         return quantize<T>(value * scale);
                                       },
                                       factor))
                  .compile();
  auto initial = device.upload<T>(seed);
  auto serial_first = device.buffer<T>(seed.size());
  auto serial_second = device.buffer<T>(seed.size());
  auto recurrent = device.buffer<T>(seed.size());
  if (!body || !initial || !serial_first || !serial_second || !recurrent) {
    return 1;
  }

  std::array<T, 4u> serial{};
  for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
    const auto &source =
        iteration == 0u
            ? *initial
            : ((iteration & 1u) != 0u ? *serial_first : *serial_second);
    auto &target = (iteration & 1u) == 0u ? *serial_first : *serial_second;
    auto result = body->run(source, target);
    if (!result || (iteration + 1u == iterations &&
                    !result->read(target, std::span<T>{serial}))) {
      return 2;
    }
  }

  auto prepared =
      pipeline(device)
          .template repeat<iterations>(*body, read(*initial), write(*recurrent))
          .prepare();
  std::array<T, 4u> actual{};
  if (!prepared) {
    std::fprintf(stderr, "%s prepare failed: %.*s\n", name,
                 static_cast<int>(prepared.error().size()),
                 prepared.error().data());
    return 3;
  }
  const Status status = prepared->run();
  if (!status) {
    std::fprintf(stderr, "%s run failed: %.*s\n", name,
                 static_cast<int>(status.error().size()),
                 status.error().data());
    return 4;
  }
  const Stats run_stats = prepared->stats();
  if (!prepared->read(*recurrent, actual) || actual != serial) {
    std::fprintf(stderr, "%s parity failed actual=%lld serial=%lld\n", name,
                 static_cast<long long>(actual[0].raw()),
                 static_cast<long long>(serial[0].raw()));
    return 5;
  }
  const Stats observed_stats = prepared->stats();
  const std::uint64_t expected_submits =
      run_stats.backend == Backend::Vulkan ? 2u : 1u;
  if (observed_stats.command_submits != expected_submits ||
      observed_stats.dispatches != 1u ||
      observed_stats.pipeline.step_count != 1u ||
      observed_stats.pipeline.verified_step_count != 1u) {
    std::fprintf(
        stderr, "%s topology failed submits=%llu dispatches=%llu\n", name,
        static_cast<unsigned long long>(observed_stats.command_submits),
        static_cast<unsigned long long>(observed_stats.dispatches));
    std::fprintf(
        stderr,
        "%s pre-read submits=%llu dispatches=%llu reads=%llu bytes=%llu\n",
        name, static_cast<unsigned long long>(run_stats.command_submits),
        static_cast<unsigned long long>(run_stats.dispatches),
        static_cast<unsigned long long>(run_stats.download_events),
        static_cast<unsigned long long>(run_stats.downloaded_bytes));
    return 6;
  }
  return 0;
}

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
  if (!prepared || !prepared->run()) {
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
  auto prepared = pipeline(device)
                      .windows<maximum, tile>(
                          *body, rund::compute::window(*count).until<1u>(7u),
                          read(*initial, *terminal), write(*output, *stopped))
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

int Check(const rund::compute::Target target, rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto first = rund::compute::on(device)
                   .map<std::int32_t>("session-accel-pipeline-double",
                                      input_values.size(),
                                      [](auto value) { return value * 2; })
                   .compile();
  auto second = rund::compute::on(device)
                    .map<std::int32_t>("session-accel-pipeline-advance",
                                       input_values.size(),
                                       [](auto value) { return value + 3; })
                    .compile();
  auto input = device.upload<std::int32_t>(input_values);
  auto middle = device.buffer<std::int32_t>(input_values.size());
  auto output = device.buffer<std::int32_t>(input_values.size());
  if (!first || !second || !input || !middle || !output) {
    return 2;
  }
  auto prepared = rund::compute::pipeline(device)
                      .then(*first, rund::compute::read(*input),
                            rund::compute::write(*middle))
                      .then(*second, rund::compute::read(*middle),
                            rund::compute::write(*output))
                      .prepare();
  if (!prepared) {
    return 3;
  }
  rund::Session session{};
  if (!session.open(rund::node::test_contract::Options())) {
    return 4;
  }
  auto submission = session.compute(*prepared).submit();
  const rund::compute::Completion completion = submission.wait();
  if (!completion) {
    return 5;
  }
  const rund::compute::Stats stats = completion.stats();
  if (stats.backend != target.backend() || stats.pipeline.step_count != 2u ||
      stats.pipeline.resource_count != 3u ||
      stats.pipeline.barrier_count != 1u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.failed_step_index !=
          rund::compute::PipelineStats::no_failed_step ||
      stats.pipeline.control_byte_count != 128u ||
      stats.pipeline.control_command_count != 2u ||
      stats.command_submits != 1u || stats.dispatches != 2u ||
      prepared->generation() != 1u) {
    std::fprintf(
        stderr,
        "pipeline topology backend=%u steps=%llu resources=%llu barriers=%llu "
        "verified=%llu failed=%llu control-bytes=%llu control-commands=%llu "
        "submits=%llu dispatches=%llu generation=%llu\n",
        static_cast<unsigned>(target.backend()),
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.resource_count),
        static_cast<unsigned long long>(stats.pipeline.barrier_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.pipeline.control_byte_count),
        static_cast<unsigned long long>(stats.pipeline.control_command_count),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(prepared->generation()));
    return 6;
  }
  std::array<std::int32_t, input_values.size()> values{};
  if (!prepared->read(*output, values) ||
      values != std::array<std::int32_t, 4u>{5, 7, 9, 11}) {
    return 7;
  }
  if (const int observed = CheckObservation(device); observed != 0) {
    return 70 + observed;
  }
  if (const int window = CheckWindow(session, device); window != 0) {
    return 75 + window;
  }
  const std::array<rund::compute::Fixed<16, 16>, 4u> seed32{
      rund::compute::Fixed<16, 16>::from_raw(65537),
      rund::compute::Fixed<16, 16>::from_raw(98305),
      rund::compute::Fixed<16, 16>::from_raw(-65537),
      rund::compute::Fixed<16, 16>::from_raw(
          std::numeric_limits<std::int32_t>::max() - 2)};
  constexpr std::int64_t one64 = std::int64_t{1} << 44u;
  const std::array<rund::compute::Fixed<20, 44>, 4u> seed64{
      rund::compute::Fixed<20, 44>::from_raw(one64 + 1),
      rund::compute::Fixed<20, 44>::from_raw(one64 + (one64 >> 1u) + 1),
      rund::compute::Fixed<20, 44>::from_raw(-(one64 + 1)),
      rund::compute::Fixed<20, 44>::from_raw(
          std::numeric_limits<std::int64_t>::max() - 2)};
  if (const int fixed = CheckFixedRecurrence(
          device, seed32, rund::compute::Fixed<16, 16>::from_raw(98305),
          "pipeline-fixed-i16-f16-recurrence");
      fixed != 0) {
    return 80 + fixed;
  }
  if (const int fixed = CheckFixedRecurrence(
          device, seed64,
          rund::compute::Fixed<20, 44>::from_raw(one64 + (one64 >> 1u) + 1),
          "pipeline-fixed-i20-f44-recurrence");
      fixed != 0) {
    return 90 + fixed;
  }
  return session.close() ? 0 : 8;
}

} // namespace

int RunRuntimeComputePipelineAccelContract() {
  for (const rund::compute::Target target :
       {rund::compute::Target::metal(), rund::compute::Target::vulkan()}) {
    auto device = rund::compute::open(target);
    if (!device) {
      if (device.reason() != rund::compute::Reason::AdapterUnavailable) {
        return static_cast<int>(target.backend()) * 10 + 1;
      }
      continue;
    }
    const int result = Check(target, *device);
    if (result != 0) {
      return static_cast<int>(target.backend()) * 10 + result;
    }
  }
  return 0;
}
