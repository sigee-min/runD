#pragma once

#include "values.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute/expr/functions/hash.hpp>
#include <rund/compute/flow/builder.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace rund::node::test_contract::expression {
namespace {

template <class Executor, class Input, std::size_t N, class Function>
[[nodiscard]] int
check_single_parity(Executor &executor, const rund::compute::Backend backend,
                    const std::array<Input, N> &input, const char *const label,
                    const Function function) {
  using namespace rund::compute;
  const auto make = [&](const Backend selected) {
    return (on(rund::node::test_contract::target_for(selected, 2u)))
        .template map<Input>(label, input.size(), function)
        .compile();
  };
  auto cpu_program = make(Backend::Cpu);
  auto target_program = make(backend);
  if (!cpu_program || !target_program) {
    const auto &failed = !cpu_program ? cpu_program : target_program;
    std::fprintf(
        stderr, "expression compile failed label=%s backend=%u reason=%.*s\n",
        label, static_cast<unsigned>(backend),
        static_cast<int>(failed.error().size()), failed.error().data());
    return 1;
  }
  auto cpu = cpu_program->resident(input);
  auto target = target_program->resident(input);
  if (!cpu || !target || !executor(*cpu) || !executor(*target)) {
    std::fprintf(stderr, "expression run failed label=%s backend=%u\n", label,
                 static_cast<unsigned>(backend));
    return 2;
  }
  auto cpu_output = cpu->read();
  auto target_output = target->read();
  const Stats cpu_stats = cpu->stats();
  const Stats target_stats = target->stats();
  const bool matches = cpu_output && target_output &&
                       *cpu_output == *target_output &&
                       cpu_stats.graph_hash == target_stats.graph_hash &&
                       cpu_stats.output_hash == target_stats.output_hash;
  if (!matches) {
    std::fprintf(stderr, "expression mismatch label=%s backend=%u\n", label,
                 static_cast<unsigned>(backend));
    if (cpu_output && target_output) {
      const std::size_t count =
          std::min(cpu_output->size(), target_output->size());
      for (std::size_t index = 0u; index < count; ++index) {
        if ((*cpu_output)[index] != (*target_output)[index]) {
          std::fprintf(
              stderr,
              "expression value mismatch index=%zu cpu=%lld target=%lld\n",
              index, printable((*cpu_output)[index]),
              printable((*target_output)[index]));
          break;
        }
      }
    }
    return 3;
  }
  return 0;
}

template <class Executor, class Input, std::size_t N, class Output,
          class Function>
[[nodiscard]] int
check_single_expected(Executor &executor, const rund::compute::Backend backend,
                      const std::array<Input, N> &input,
                      const std::array<Output, N> &expected,
                      const char *const label, const Function function) {
  using namespace rund::compute;
  auto program = (on(rund::node::test_contract::target_for(backend, 2u)))
                     .template map<Input>(label, input.size(), function)
                     .compile();
  if (!program) {
    std::fprintf(stderr,
                 "expression golden compile failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr,
                 "expression golden resident failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  const auto run = executor(*job);
  if (!run) {
    std::fprintf(stderr,
                 "expression golden run failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(run.error().size()), run.error().data());
    return 3;
  }
  auto output = job->read();
  const std::vector<Output> expected_values(expected.begin(), expected.end());
  const Stats stats = job->stats();
  if (!output || *output != expected_values || stats.graph_hash == 0u ||
      stats.output_hash == 0u) {
    std::fprintf(stderr,
                 "expression golden mismatch label=%s backend=%u reason=%.*s "
                 "graph=%llu output=%llu\n",
                 label, static_cast<unsigned>(backend),
                 output ? 0 : static_cast<int>(output.error().size()),
                 output ? "" : output.error().data(),
                 static_cast<unsigned long long>(stats.graph_hash),
                 static_cast<unsigned long long>(stats.output_hash));
    if (output) {
      const std::size_t count =
          std::min(output->size(), expected_values.size());
      for (std::size_t index = 0u; index < count; ++index) {
        if ((*output)[index] != expected_values[index]) {
          std::fprintf(stderr,
                       "expression golden value mismatch index=%zu got=%lld "
                       "expected=%lld\n",
                       index, printable((*output)[index]),
                       printable(expected_values[index]));
          break;
        }
      }
    }
    return 4;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
