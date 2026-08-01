#include "model.hpp"

namespace rund::measure::compute {

void PrintRecurrenceColumns() {
  std::fputs(
      "recurrence_columns,backend,command_path,status,count,iterations,"
      "samples,serial_first,recurrence_first,serial_wall_median_us,"
      "recurrence_wall_median_us,speedup,history_pair_terminal_first,"
      "history_first,history_pair_terminal_wall_median_us,"
      "history_wall_median_us,history_cost_ratio,serial_command_submits,"
      "recurrence_command_submits,serial_dispatches,recurrence_dispatches,"
      "history_command_submits,history_dispatches,"
      "step_count,barrier_count,verified_step_count,serial_content_hash,"
      "recurrence_content_hash,content_parity,history_content_hash,"
      "history_exact,warm_pipeline_compiles,"
      "warm_buffer_allocations,warm_descriptor_pool_creations,"
      "warm_descriptor_set_allocations,warm_uploaded_bytes,"
      "warm_download_events,warm_downloaded_bytes,"
      "warm_internal_roundtrip_bytes,warm_external_roundtrip_bytes,"
      "warm_zero\n",
      stdout);
}

bool MeasureRecurrence(const Backend backend, const std::size_t count,
                       const std::size_t samples) {
  constexpr std::size_t iterations = 256u;
  if ((backend != Backend::Metal && backend != Backend::Vulkan) ||
      count == 0u ||
      count > std::numeric_limits<std::size_t>::max() / iterations ||
      samples == 0u || samples % 2u != 0u) {
    std::fprintf(stderr, "recurrence measurement configuration invalid\n");
    return false;
  }

  std::vector<std::int32_t> seed_values(count);
  for (std::size_t index = 0u; index < count; ++index) {
    seed_values[index] = static_cast<std::int32_t>(index % 127u) - 63;
  }
  auto device = ::rund::compute::open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "recurrence %s open failed: %.*s\n", Name(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  auto body = ::rund::compute::on(*device)
                  .map<std::int32_t>("measure-recurrence", count,
                                     [](auto value) { return value + 1; })
                  .compile();
  auto seed = device->upload<std::int32_t>(seed_values);
  auto serial_first_buffer = device->buffer<std::int32_t>(count);
  auto serial_second_buffer = device->buffer<std::int32_t>(count);
  auto recurrence_output = device->buffer<std::int32_t>(count);
  auto history_output = device->buffer<std::int32_t>(
      count * static_cast<std::size_t>(iterations));
  if (!body || !seed || !serial_first_buffer || !serial_second_buffer ||
      !recurrence_output || !history_output) {
    std::fprintf(stderr, "recurrence %s preparation input failed\n",
                 Name(backend));
    return false;
  }
  auto prepared =
      ::rund::compute::pipeline(*device)
          .repeat<iterations>(*body, ::rund::compute::read(*seed),
                              ::rund::compute::write_final(*recurrence_output))
          .prepare();
  if (!prepared) {
    std::fprintf(stderr, "recurrence %s prepare failed: %.*s\n", Name(backend),
                 static_cast<int>(prepared.error().size()),
                 prepared.error().data());
    return false;
  }
  auto history_prepared =
      ::rund::compute::pipeline(*device)
          .repeat<iterations>(*body, ::rund::compute::read(*seed),
                              ::rund::compute::write_each(*history_output))
          .prepare();
  if (!history_prepared) {
    std::fprintf(stderr, "recurrence %s history prepare failed: %.*s\n",
                 Name(backend),
                 static_cast<int>(history_prepared.error().size()),
                 history_prepared.error().data());
    return false;
  }

  std::vector<double> serial_wall;
  std::vector<double> recurrence_wall;
  std::vector<double> history_pair_terminal_wall;
  std::vector<double> history_wall;
  serial_wall.reserve(samples);
  recurrence_wall.reserve(samples);
  history_pair_terminal_wall.reserve(samples);
  history_wall.reserve(samples);
  WarmCounters warm{};
  ExecutionCounters serial_counters{};
  ExecutionCounters recurrence_counters{};
  ExecutionCounters history_counters{};
  Stats recurrence_stats{};
  Stats history_stats{};
  std::vector<std::int32_t> serial_values(count);
  std::vector<std::int32_t> recurrence_values(count);
  std::vector<std::int32_t> history_values(count * iterations);

  const auto run_serial = [&](const bool timed, const bool observe) {
    std::uint64_t submits = 0u;
    std::uint64_t dispatches = 0u;
    const auto begin = Clock::now();
    for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
      const auto &source =
          iteration == 0u ? *seed
                          : ((iteration & 1u) != 0u ? *serial_first_buffer
                                                    : *serial_second_buffer);
      auto &target =
          (iteration & 1u) == 0u ? *serial_first_buffer : *serial_second_buffer;
      auto result = body->run(source, target);
      if (!result) {
        std::fprintf(stderr, "recurrence %s serial execution failed\n",
                     Name(backend));
        return false;
      }
      const Stats stats = result->stats();
      ObserveWarm(warm, stats);
      ::rund::detail::counter::Accumulate(submits, stats.command_submits);
      ::rund::detail::counter::Accumulate(dispatches, stats.dispatches);
      if (observe && iteration + 1u == iterations) {
        const auto status =
            result->read(target, std::span<std::int32_t>{serial_values});
        if (!status) {
          std::fprintf(stderr, "recurrence %s serial read failed\n",
                       Name(backend));
          return false;
        }
      }
    }
    const auto end = Clock::now();
    if (submits != iterations || dispatches != iterations) {
      std::fprintf(stderr, "recurrence %s serial counters failed\n",
                   Name(backend));
      return false;
    }
    serial_counters = {.command_submits = submits, .dispatches = dispatches};
    if (timed) {
      serial_wall.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return true;
  };

  const auto run_recurrence = [&](std::vector<double> *const wall,
                                  const bool observe) {
    const auto begin = Clock::now();
    const auto status = prepared->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "recurrence %s execution failed: %.*s\n",
                   Name(backend), static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    recurrence_stats = prepared->stats();
    ObserveWarm(warm, recurrence_stats);
    if (recurrence_stats.command_submits != 1u ||
        recurrence_stats.dispatches != 1u ||
        recurrence_stats.pipeline.step_count != 1u ||
        recurrence_stats.pipeline.verified_step_count != 1u) {
      std::fprintf(stderr, "recurrence %s counters failed\n", Name(backend));
      return false;
    }
    recurrence_counters = {
        .command_submits = recurrence_stats.command_submits,
        .dispatches = recurrence_stats.dispatches,
    };
    if (wall != nullptr) {
      wall->push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    if (observe) {
      const auto read = prepared->read(
          *recurrence_output, std::span<std::int32_t>{recurrence_values});
      if (!read) {
        std::fprintf(stderr, "recurrence %s read failed\n", Name(backend));
        return false;
      }
    }
    return true;
  };

  const auto run_history = [&](std::vector<double> *const wall,
                               const bool observe) {
    const auto begin = Clock::now();
    const auto status = history_prepared->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "recurrence %s history execution failed: %.*s\n",
                   Name(backend), static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    history_stats = history_prepared->stats();
    ObserveWarm(warm, history_stats);
    if (history_stats.command_submits != 1u || history_stats.dispatches != 1u ||
        history_stats.pipeline.step_count != 1u ||
        history_stats.pipeline.verified_step_count != 1u) {
      std::fprintf(stderr, "recurrence %s history counters failed\n",
                   Name(backend));
      return false;
    }
    history_counters = {
        .command_submits = history_stats.command_submits,
        .dispatches = history_stats.dispatches,
    };
    if (wall != nullptr) {
      wall->push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    if (observe) {
      const auto read = history_prepared->read(
          *history_output, std::span<std::int32_t>{history_values});
      if (!read) {
        std::fprintf(stderr, "recurrence %s history read failed\n",
                     Name(backend));
        return false;
      }
    }
    return true;
  };

  if (!run_serial(false, false) || !run_recurrence(nullptr, false) ||
      !run_history(nullptr, false)) {
    return false;
  }
  std::size_t serial_first{};
  std::size_t recurrence_first{};
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_then_recurrence = sample % 2u == 0u;
    const bool ok =
        serial_then_recurrence
            ? run_serial(true, false) && run_recurrence(&recurrence_wall, false)
            : run_recurrence(&recurrence_wall, false) &&
                  run_serial(true, false);
    if (!ok) {
      return false;
    }
    serial_first += serial_then_recurrence ? 1u : 0u;
    recurrence_first += serial_then_recurrence ? 0u : 1u;
  }
  std::size_t history_pair_terminal_first{};
  std::size_t history_first{};
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool terminal_then_history = sample % 2u == 0u;
    const bool ok =
        terminal_then_history
            ? run_recurrence(&history_pair_terminal_wall, false) &&
                  run_history(&history_wall, false)
            : run_history(&history_wall, false) &&
                  run_recurrence(&history_pair_terminal_wall, false);
    if (!ok) {
      return false;
    }
    history_pair_terminal_first += terminal_then_history ? 1u : 0u;
    history_first += terminal_then_history ? 0u : 1u;
  }
  if (!run_serial(false, true) || !run_recurrence(nullptr, true) ||
      !run_history(nullptr, true)) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const std::int32_t expected =
        seed_values[index] + static_cast<std::int32_t>(iterations);
    if (serial_values[index] != expected ||
        recurrence_values[index] != expected) {
      std::fprintf(stderr, "recurrence %s content failed\n", Name(backend));
      return false;
    }
  }
  bool history_exact = true;
  for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
    for (std::size_t index = 0u; index < count; ++index) {
      const std::int32_t expected =
          seed_values[index] + static_cast<std::int32_t>(iteration + 1u);
      if (history_values[iteration * count + index] != expected) {
        history_exact = false;
        break;
      }
    }
    if (!history_exact) {
      break;
    }
  }
  if (!history_exact) {
    std::fprintf(stderr, "recurrence %s history content failed\n",
                 Name(backend));
    return false;
  }

  const std::uint64_t serial_hash = ContentHash(serial_values);
  const std::uint64_t recurrence_hash = ContentHash(recurrence_values);
  const std::uint64_t history_hash = ContentHash(history_values);
  const bool balanced =
      serial_first == samples / 2u && recurrence_first == samples / 2u;
  const bool history_balanced = history_pair_terminal_first == samples / 2u &&
                                history_first == samples / 2u;
  const bool parity = serial_hash != 0u && serial_hash == recurrence_hash;
  const bool contract = balanced && history_balanced && parity &&
                        history_exact && history_hash != 0u &&
                        history_stats.pipeline.barrier_count ==
                            recurrence_stats.pipeline.barrier_count &&
                        warm.zero();
  const double serial_us = Median(serial_wall);
  const double recurrence_us = Median(recurrence_wall);
  const double speedup = recurrence_us == 0.0 ? 0.0 : serial_us / recurrence_us;
  const double history_pair_terminal_us = Median(history_pair_terminal_wall);
  const double history_us = Median(history_wall);
  const double history_cost_ratio = history_pair_terminal_us == 0.0
                                        ? 0.0
                                        : history_us / history_pair_terminal_us;
  const auto &evidence = recurrence_stats.pipeline;
  std::printf(
      "recurrence,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%.3f,%.3f,%.6f,"
      "%zu,%zu,%.3f,%.3f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%u,%llu,%u,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%u\n",
      Name(backend), CommandPath(backend), contract ? "ok" : "contract_failed",
      count, iterations, samples, serial_first, recurrence_first, serial_us,
      recurrence_us, speedup, history_pair_terminal_first, history_first,
      history_pair_terminal_us, history_us, history_cost_ratio,
      static_cast<unsigned long long>(serial_counters.command_submits),
      static_cast<unsigned long long>(recurrence_counters.command_submits),
      static_cast<unsigned long long>(serial_counters.dispatches),
      static_cast<unsigned long long>(recurrence_counters.dispatches),
      static_cast<unsigned long long>(history_counters.command_submits),
      static_cast<unsigned long long>(history_counters.dispatches),
      static_cast<unsigned long long>(evidence.step_count),
      static_cast<unsigned long long>(evidence.barrier_count),
      static_cast<unsigned long long>(evidence.verified_step_count),
      static_cast<unsigned long long>(serial_hash),
      static_cast<unsigned long long>(recurrence_hash), parity ? 1u : 0u,
      static_cast<unsigned long long>(history_hash), history_exact ? 1u : 0u,
      static_cast<unsigned long long>(warm.pipeline_compiles),
      static_cast<unsigned long long>(warm.buffer_allocations),
      static_cast<unsigned long long>(warm.descriptor_pool_creations),
      static_cast<unsigned long long>(warm.descriptor_set_allocations),
      static_cast<unsigned long long>(warm.uploaded_bytes),
      static_cast<unsigned long long>(warm.download_events),
      static_cast<unsigned long long>(warm.downloaded_bytes),
      static_cast<unsigned long long>(warm.internal_roundtrip_bytes),
      static_cast<unsigned long long>(warm.external_roundtrip_bytes),
      warm.zero() ? 1u : 0u);
  return contract;
}

} // namespace rund::measure::compute
