#pragma once

#include "values.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute/expr/functions/hash.hpp>
#include <rund/compute/flow/builder.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <type_traits>
#include <utility>

namespace rund::node::test_contract::expression {
namespace {

template <class Executor, class Input, std::size_t N, class Function>
[[nodiscard]] int
check_record_parity(Executor &executor, const rund::compute::Backend backend,
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
  if (!cpu || !target) {
    const auto &failed = !cpu ? cpu : target;
    std::fprintf(
        stderr, "expression resident failed label=%s backend=%u reason=%.*s\n",
        label, static_cast<unsigned>(backend),
        static_cast<int>(failed.error().size()), failed.error().data());
    return 2;
  }
  const auto cpu_run = executor(*cpu);
  const auto target_run = executor(*target);
  if (!cpu_run || !target_run) {
    const auto &failed = !cpu_run ? cpu_run : target_run;
    std::fprintf(
        stderr, "expression run failed label=%s backend=%u reason=%.*s\n",
        label, static_cast<unsigned>(backend),
        static_cast<int>(failed.error().size()), failed.error().data());
    return 3;
  }
  auto cpu_output = cpu->read_all();
  auto target_output = target->read_all();
  if (!cpu_output || !target_output || *cpu_output != *target_output) {
    std::fprintf(stderr, "expression output mismatch label=%s backend=%u\n",
                 label, static_cast<unsigned>(backend));
    if (cpu_output && target_output) {
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        (([&] {
           const auto &left = std::get<I>(*cpu_output);
           const auto &right = std::get<I>(*target_output);
           if (left == right) {
             return;
           }
           const std::size_t count = std::min(left.size(), right.size());
           for (std::size_t index = 0u; index < count; ++index) {
             if (left[index] != right[index]) {
               std::fprintf(stderr,
                            "expression field mismatch field=%zu index=%zu "
                            "cpu=%lld target=%lld\n",
                            I, index, printable(left[index]),
                            printable(right[index]));
               return;
             }
           }
           std::fprintf(stderr,
                        "expression field size mismatch field=%zu cpu=%zu "
                        "target=%zu\n",
                        I, left.size(), right.size());
         }()),
         ...);
      }(std::make_index_sequence<
          std::tuple_size_v<std::remove_cvref_t<decltype(*cpu_output)>>>{});
    }
    return 4;
  }
  const Stats cpu_stats = cpu->stats();
  const Stats target_stats = target->stats();
  if (cpu_stats.graph_hash == 0u || cpu_stats.output_hash == 0u ||
      cpu_stats.graph_hash != target_stats.graph_hash ||
      cpu_stats.output_hash != target_stats.output_hash) {
    std::fprintf(stderr,
                 "expression evidence mismatch label=%s backend=%u "
                 "graph=%llu/%llu output=%llu/%llu\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(cpu_stats.graph_hash),
                 static_cast<unsigned long long>(target_stats.graph_hash),
                 static_cast<unsigned long long>(cpu_stats.output_hash),
                 static_cast<unsigned long long>(target_stats.output_hash));
    return 5;
  }
  return 0;
}

template <class Executor, class Input, std::size_t N, class Output,
          std::size_t FieldCount, class Function>
[[nodiscard]] int check_record_expected(
    Executor &executor, const rund::compute::Backend backend,
    const std::array<Input, N> &input,
    const std::array<std::array<Output, N>, FieldCount> &expected,
    const char *const label, const Function function) {
  using namespace rund::compute;
  auto program = (on(rund::node::test_contract::target_for(backend, 2u)))
                     .template map<Input>(label, input.size(), function)
                     .compile();
  if (!program) {
    std::fprintf(stderr,
                 "expression expected compile failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !executor(*job)) {
    std::fprintf(stderr, "expression expected run failed label=%s backend=%u\n",
                 label, static_cast<unsigned>(backend));
    return 2;
  }
  auto output = job->read_all();
  if (!output) {
    return 3;
  }
  constexpr std::size_t actual_fields =
      std::tuple_size_v<std::remove_cvref_t<decltype(*output)>>;
  static_assert(actual_fields == FieldCount);
  bool matches = true;
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((matches =
          matches && std::get<I>(*output).size() == N &&
          std::equal(std::get<I>(*output).begin(), std::get<I>(*output).end(),
                     expected[I].begin(), expected[I].end())),
     ...);
  }(std::make_index_sequence<FieldCount>{});
  if (!matches) {
    std::fprintf(stderr,
                 "expression expected output mismatch label=%s backend=%u\n",
                 label, static_cast<unsigned>(backend));
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (([&] {
         const auto &actual = std::get<I>(*output);
         if (actual.size() != N) {
           std::fprintf(stderr,
                        "expression expected field size mismatch field=%zu "
                        "actual=%zu expected=%zu\n",
                        I, actual.size(), N);
           return;
         }
         for (std::size_t index = 0u; index < N; ++index) {
           if (actual[index] != expected[I][index]) {
             std::fprintf(stderr,
                          "expression expected field mismatch field=%zu "
                          "index=%zu actual=%lld expected=%lld\n",
                          I, index, printable(actual[index]),
                          printable(expected[I][index]));
             return;
           }
         }
       }()),
       ...);
    }(std::make_index_sequence<FieldCount>{});
    return 4;
  }
  const Stats stats = job->stats();
  if (stats.graph_hash == 0u || stats.output_hash == 0u) {
    return 5;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
