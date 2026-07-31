#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <tuple>
#include <vector>

namespace {

struct Left final {};
struct Right final {};
struct Sum final {};
struct Values final {};

[[nodiscard]] auto Target() {
  return rund::compute::on(rund::compute::Target::cpu(2u));
}

[[nodiscard]] int CheckOutputs() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  const std::vector<std::int32_t> doubled{2, 4, 6, 8};
  const std::vector<std::int32_t> incremented{3, 5, 7, 9};

  auto program = Target()
                     .map<std::int32_t>("double", input.size(),
                                        [](auto value) { return value * 2; })
                     .branch([](auto values) {
                       auto next = values.map(
                           "increment", [](auto value) { return value + 1; });
                       return outputs(values, next, values);
                     })
                     .compile();
  if (!program || program->output_count() != 3u ||
      program->template output_size<0u>() != input.size() ||
      program->template output_size<1u>() != input.size() ||
      program->template output_size<2u>() != input.size()) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run() || job->stats().download_events != 0u ||
      job->stats().output_hash != 0u) {
    return 2;
  }
  auto second = job->template read<1u>();
  auto first = job->template read<0u>();
  auto third = job->template read<2u>();
  const std::uint64_t ordered_hash = job->stats().output_hash;
  if (!first || !second || !third || *first != doubled ||
      *second != incremented || *third != doubled || ordered_hash == 0u ||
      job->stats().download_events != 3u) {
    return 3;
  }

  auto all_job = program->resident(input);
  if (!all_job || !all_job->run()) {
    return 4;
  }
  auto all = all_job->read_all();
  if (!all || std::get<0>(*all) != doubled ||
      std::get<1>(*all) != incremented || std::get<2>(*all) != doubled ||
      all_job->stats().output_hash != ordered_hash) {
    return 5;
  }

  auto reversed = Target()
                      .map<std::int32_t>("double", input.size(),
                                         [](auto value) { return value * 2; })
                      .branch([](auto values) {
                        auto next = values.map(
                            "increment", [](auto value) { return value + 1; });
                        return outputs(next, values, values);
                      })
                      .compile();
  if (!reversed) {
    return 6;
  }
  auto reversed_job = reversed->resident(input);
  if (!reversed_job || !reversed_job->run() ||
      reversed_job->stats().graph_hash == job->stats().graph_hash) {
    return 7;
  }
  auto reversed_output = reversed_job->read_all();
  return reversed_output && std::get<0>(*reversed_output) == incremented &&
                 std::get<1>(*reversed_output) == doubled &&
                 std::get<2>(*reversed_output) == doubled
             ? 0
             : 8;
}

[[nodiscard]] int CheckIdentityProjection() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  const std::vector<std::int32_t> expected{2, 4, 6, 8};
  auto program =
      Target()
          .map<std::int32_t>("identity-source", input.size(),
                             [](auto value) { return value * 2; })
          .branch([](auto values) {
            const auto identity = values.map(
                "identity-output", [](auto value) { return value; });
            return outputs(values, identity);
          })
          .compile();
  if (!program) {
    std::fprintf(stderr, "identity projection compile reason=%.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  if (program->output_count() != 2u ||
      program->template output_size<0u>() != input.size() ||
      program->template output_size<1u>() != input.size() ||
      program->graph().nodes.size() != 1u ||
      program->graph().outputs.size() != 1u) {
    std::fprintf(stderr,
                 "identity projection shape outputs=%zu sizes=%zu/%zu "
                 "nodes=%zu physical=%zu\n",
                 program->output_count(), program->template output_size<0u>(),
                 program->template output_size<1u>(),
                 program->graph().nodes.size(),
                 program->graph().outputs.size());
    return 1;
  }
  auto result = program->run(input);
  return result && std::get<0>(*result) == expected &&
                 std::get<1>(*result) == expected
             ? 0
             : 2;
}

[[nodiscard]] int CheckComposition() {
  using namespace rund::compute;
  const std::array<std::uint64_t, 4u> input{3u, 1u, 4u, 2u};
  const std::array<std::uint64_t, 4u> side{4u, 3u, 2u, 1u};

  auto result =
      on(Target::cpu(2u), input)
          .combine("pair-sum", side,
                   [](auto left, auto right) { return left + right; })
          .pipe([](auto values) {
            return values.map("double", [](auto value) { return value * 2u; });
          })
          .branch([](auto values) {
            auto repeated = values.template unroll<2u>([](auto stage) {
              return stage.map("increment",
                               [](auto value) { return value + 1u; });
            });
            auto total = repeated.reduce(Reduce::Sum);
            auto fields = record(field<Values>(repeated), field<Sum>(total));
            auto adjusted = fields.template get<Values>().combine(
                "add-total", fields.template get<Sum>(),
                [](auto value, auto sum) { return value + sum; });
            return outputs(fields, adjusted, values.indices());
          })
          .collect();
  if (!result) {
    std::fprintf(stderr, "flow composition reason=%.*s\n",
                 static_cast<int>(result.error().size()),
                 result.error().data());
    return 1;
  }
  const auto &fields = std::get<0>(*result);
  return std::get<0>(fields) == std::vector<std::uint64_t>{16u, 10u, 14u, 8u} &&
                 std::get<1>(fields) == 48u &&
                 std::get<1>(*result) ==
                     std::vector<std::uint64_t>{64u, 58u, 62u, 56u} &&
                 std::get<2>(*result) ==
                     std::vector<std::uint64_t>{0u, 1u, 2u, 3u}
             ? 0
             : 2;
}

[[nodiscard]] int CheckBoundedPipe() {
  using namespace rund::compute;
  const std::array<std::uint32_t, 4u> input{3u, 1u, 4u, 2u};
  const auto normalize = [](auto values) {
    auto total = values.reduce(Reduce::Sum);
    return values.combine("normalize", total, [](auto value, auto sum) {
      return value * 90u / sum;
    });
  };
  auto output = on(Target::cpu(2u), input)
                    .filter([](auto value) { return value > 1u; })
                    .pipe(normalize)
                    .collect();
  return output && *output == std::vector<std::uint32_t>{30u, 40u, 20u} ? 0 : 1;
}

[[nodiscard]] int CheckBoundedGather() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> source{10, 20, 30, 40};
  const std::array<std::uint32_t, 4u> indices{2u, 99u, 1u, 0u};
  auto program =
      on(Target::cpu(2u))
          .input<std::int32_t>(source.size())
          .zip_input<std::uint32_t>(indices.size())
          .branch([](auto values, auto requested) {
            auto active = requested.filter(
                [](auto index) { return index != std::uint32_t{99}; });
            return values.gather(active);
          })
          .compile();
  if (!program) {
    return 1;
  }
  const auto backend = program->backend();
  if (!backend || *backend != Backend::Cpu) return 1;
  const auto fingerprint = program->graph().fingerprint;
  const std::size_t nodes = program->graph().nodes.size();
  const std::size_t resources = program->graph().resources.size();
  auto gathered = program->run(source, indices);
  if (!gathered || *gathered != std::vector<std::int32_t>{30, 20, 10}) {
    return 2;
  }
  const std::array<std::uint32_t, 4u> invalid{2u, 9u, 1u, 0u};
  auto rejected = program->run(source, invalid);
  return !rejected &&
                 rejected.error() == "compute_gather_index_out_of_range" &&
                 *backend == Backend::Cpu &&
                 program->graph().fingerprint == fingerprint &&
                 program->graph().nodes.size() == nodes &&
                 program->graph().resources.size() == resources
             ? 0
             : 3;
}

[[nodiscard]] int CheckIndexedMap() {
  using namespace rund::compute;
  const std::array<std::int32_t, 6u> source{10, 20, 30, 40, 50, 60};
  const std::array<std::uint32_t, 4u> indices{5u, 1u, 3u, 0u};
  auto program =
      on(Target::cpu(2u))
          .input<std::int32_t>(source.size())
          .zip_input<std::uint32_t>(indices.size())
          .branch([](auto values, auto requested) {
            return values.gather(requested).map(
                "indexed-map", [](auto value) { return value + 7; });
          })
          .compile();
  if (!program) {
    return 1;
  }
  std::size_t maps = 0u;
  std::size_t gathers = 0u;
  for (const graph::Node &node : program->graph().nodes) {
    maps += node.operation == graph::Operation::Map ? 1u : 0u;
    gathers += node.operation == graph::Operation::Gather ? 1u : 0u;
  }
  if (maps != 1u || gathers != 0u || program->graph().nodes.size() != 1u) {
    return 2;
  }
  auto output = program->run(source, indices);
  if (!output ||
      *output != std::vector<std::int32_t>{67, 27, 47, 17}) {
    return 3;
  }
  const std::array<std::uint32_t, 4u> invalid{5u, 6u, 3u, 0u};
  auto rejected = program->run(source, invalid);
  return !rejected &&
                 rejected.error() == "compute_gather_index_out_of_range"
             ? 0
             : 4;
}

[[nodiscard]] int CheckScatterReduce() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> values{5, 7, 11, 13};
  const std::array<std::uint32_t, 4u> indices{0u, 1u, 0u, 1u};
  auto exact = on(Target::cpu(2u))
                   .input<std::int32_t>(values.size())
                   .zip_input<std::uint32_t>(indices.size())
                   .branch([](auto input, auto targets) {
                     return input.scatter_reduce(targets, 2u, Reduce::Sum);
                   })
                   .compile();
  if (!exact) return 1;
  const auto exact_backend = exact->backend();
  if (!exact_backend || *exact_backend != Backend::Cpu) return 1;
  auto reduced = exact->run(values, indices);
  if (!reduced || *reduced != std::vector<std::int32_t>{16, 20}) return 2;

  const std::array<std::uint32_t, 4u> invalid_indices{0u, 2u, 0u, 1u};
  auto invalid = exact->run(values, invalid_indices);
  if (invalid ||
      invalid.error() != "compute_scatter_reduce_index_out_of_range" ||
      *exact_backend != Backend::Cpu) {
    return 3;
  }

  auto bounded =
      on(Target::cpu(2u))
          .input<Bounded<std::int32_t>>(values.size())
          .branch([](auto input) {
            auto targets = input.indices().map(
                "scatter-reduce-target", [](auto ordinal) {
                  return ordinal & std::uint32_t{1};
                });
            return input.scatter_reduce(targets, 2u, Reduce::Sum);
          })
          .compile();
  if (!bounded) return 4;
  const std::array<std::uint32_t, 1u> overflow_count{5u};
  auto overflow = bounded->run(values, overflow_count);
  const auto bounded_backend = bounded->backend();
  if (overflow || overflow.error() != "compute_workset_overflow" ||
      !bounded_backend || *bounded_backend != Backend::Cpu) {
    return 5;
  }

  auto bounded_sort =
      on(Target::cpu(2u))
          .input<Bounded<std::int32_t>>(values.size())
          .branch([](auto input) { return input.sort(); })
          .compile();
  if (!bounded_sort) return 6;
  const auto fingerprint = bounded_sort->graph().fingerprint;
  const std::size_t nodes = bounded_sort->graph().nodes.size();
  const std::size_t resources = bounded_sort->graph().resources.size();
  auto sort_overflow = bounded_sort->run(values, overflow_count);
  if (sort_overflow ||
      sort_overflow.error() != "compute_workset_overflow" ||
      bounded_sort->graph().fingerprint != fingerprint ||
      bounded_sort->graph().nodes.size() != nodes ||
      bounded_sort->graph().resources.size() != resources) {
    return 7;
  }
  constexpr std::array<std::uint32_t, 1u> full_count{values.size()};
  auto sorted = bounded_sort->run(values, full_count);
  return sorted && *sorted == std::vector<std::int32_t>{5, 7, 11, 13}
             ? 0
             : 8;
}

[[nodiscard]] int CheckBoundedCollectiveRepeat() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> input{3u, 0u, 1u, 2u};
  auto program =
      on(Target::cpu(2u))
          .input<std::uint32_t>(input.size())
          .branch([](auto values) {
            auto active =
                values.filter([](auto value) { return value != 0u; });
            return active.template unroll<2u>(
                [](auto work) { return work.sort(); },
                [](auto value) { return value == std::uint32_t{99}; });
          })
          .compile();
  if (!program) {
    std::fprintf(stderr, "bounded collective repeat reason=%.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) return 2;
  const Status ran = job->run();
  if (!ran) return 3;
  const Stats stats = job->stats();
  auto output = job->read();
  if (!output || *output != std::vector<std::uint32_t>{1u, 2u, 3u} ||
      stats.backend != Backend::Cpu ||
      stats.control.iteration_count != 2u ||
      stats.control.skipped_iteration_count != 0u) {
    return 4;
  }

  auto scan_program =
      on(rund::compute::Target::cpu(2u))
          .input<std::uint32_t>(input.size())
          .branch([](auto values) {
            auto active =
                values.filter([](auto value) { return value != 0u; });
            return active.template unroll<2u>(
                [](auto work) { return work.scan(Scan::InclusiveSum); },
                [](auto value) { return value == std::uint32_t{99}; });
          })
          .compile();
  if (!scan_program) return 5;
  auto scan_job = scan_program->resident(input);
  if (!scan_job || !scan_job->run()) return 6;
  const Stats scan_stats = scan_job->stats();
  auto scan_output = scan_job->read();
  return scan_output &&
                 *scan_output == std::vector<std::uint32_t>{3u, 7u, 13u} &&
                 scan_stats.control.iteration_count == 2u &&
                 scan_stats.control.skipped_iteration_count == 0u
             ? 0
             : 7;
}

[[nodiscard]] int CheckBoundedBoundaryConsumers() {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> values{5, 7, 11, 13};
  constexpr std::array<std::uint32_t, 1u> overflow_count{5u};
  constexpr std::array<std::uint32_t, 1u> full_count{values.size()};

  auto scan = on(Target::cpu(2u))
                  .input<Bounded<std::int32_t>>(values.size())
                  .branch([](auto input) {
                    return input.scan(Scan::InclusiveSum);
                  })
                  .compile();
  if (!scan) return 1;
  const auto scan_fingerprint = scan->graph().fingerprint;
  const std::size_t scan_nodes = scan->graph().nodes.size();
  const std::size_t scan_resources = scan->graph().resources.size();
  auto rejected_scan = scan->run(values, overflow_count);
  if (rejected_scan ||
      rejected_scan.error() != "compute_workset_overflow" ||
      scan->graph().fingerprint != scan_fingerprint ||
      scan->graph().nodes.size() != scan_nodes ||
      scan->graph().resources.size() != scan_resources) {
    return 2;
  }
  auto scanned = scan->run(values, full_count);
  if (!scanned || *scanned != std::vector<std::int32_t>{5, 12, 23, 36}) {
    return 3;
  }
  auto scan_job = scan->resident(values, full_count);
  if (!scan_job || !scan_job->run()) return 4;
  const Status rejected_scan_write = scan_job->write(values, overflow_count);
  auto retained_scan = scan_job->read();
  if (rejected_scan_write ||
      rejected_scan_write.error() != "compute_workset_overflow" ||
      !retained_scan ||
      *retained_scan != std::vector<std::int32_t>{5, 12, 23, 36} ||
      scan->graph().fingerprint != scan_fingerprint ||
      scan->graph().nodes.size() != scan_nodes ||
      scan->graph().resources.size() != scan_resources ||
      !scan_job->write(values, full_count) || !scan_job->run()) {
    return 5;
  }

  constexpr std::array<std::uint32_t, 4u> targets{0u, 1u, 0u, 1u};
  auto scatter_reduce =
      on(Target::cpu(2u))
          .input<Bounded<std::uint32_t>>(targets.size())
          .branch([](auto input) {
            return input.scatter_reduce(input, 2u, Reduce::Sum);
          })
          .compile();
  if (!scatter_reduce) return 6;
  const auto scatter_fingerprint = scatter_reduce->graph().fingerprint;
  const std::size_t scatter_nodes = scatter_reduce->graph().nodes.size();
  const std::size_t scatter_resources =
      scatter_reduce->graph().resources.size();
  auto rejected_scatter = scatter_reduce->run(targets, overflow_count);
  if (rejected_scatter ||
      rejected_scatter.error() != "compute_workset_overflow" ||
      scatter_reduce->graph().fingerprint != scatter_fingerprint ||
      scatter_reduce->graph().nodes.size() != scatter_nodes ||
      scatter_reduce->graph().resources.size() != scatter_resources) {
    return 7;
  }
  auto scattered = scatter_reduce->run(targets, full_count);
  return scattered && *scattered == std::vector<std::uint32_t>{0u, 2u}
             ? 0
             : 8;
}

[[nodiscard]] int CheckGroup() {
  using namespace rund::compute;
  const std::array<std::int32_t, 5u> input{-1, 2, -1, 0, -2};
  auto output =
      on(Target::cpu(2u), input)
          .branch([](auto values) {
            auto groups = values.group_by([](auto value) { return value; });
            auto aggregate = groups.aggregate([](auto group) {
              return record(group.key(), group.count(),
                            group.values().reduce(Reduce::Sum));
            });
            auto ordered =
                groups.values()
                    .map("group-double", [](auto value) { return value * 2; })
                    .scan(Scan::InclusiveSum)
                    .ordered();
            return outputs(aggregate, ordered);
          })
          .collect();
  if (!output) {
    std::fprintf(stderr, "flow group reason=%.*s\n",
                 static_cast<int>(output.error().size()),
                 output.error().data());
    return 1;
  }
  const auto &groups = std::get<0>(*output);
  return std::get<0>(groups) == std::vector<std::int32_t>{-2, -1, 0, 2} &&
                 std::get<1>(groups) ==
                     std::vector<std::uint32_t>{1u, 2u, 1u, 1u} &&
                 std::get<2>(groups) ==
                     std::vector<std::int32_t>{-2, -2, 0, 2} &&
                 std::get<1>(*output) ==
                     std::vector<std::int32_t>{-4, -2, -4, 0, 4}
             ? 0
             : 2;
}

[[nodiscard]] int CheckJoin() {
  using namespace rund::compute;
  const std::array<std::uint32_t, 3u> left{10u, 21u, 12u};
  const std::array<std::uint32_t, 4u> right{42u, 14u, 33u, 26u};
  auto output =
      on(Target::cpu(2u), left)
          .join(
              MaxMatches{3u}, right, [](auto value) { return value & 1u; },
              [](auto value) { return value & 1u; },
              [](auto first, auto second) {
                return record(field<Left>(first), field<Right>(second));
              })
          .collect();
  if (!output ||
      std::get<0>(*output) !=
          std::vector<std::uint32_t>{10u, 10u, 10u, 12u, 12u, 12u, 21u} ||
      std::get<1>(*output) !=
          std::vector<std::uint32_t>{42u, 14u, 26u, 42u, 14u, 26u, 33u}) {
    return 1;
  }

  const std::array<std::uint32_t, 1u> one{2u};
  const std::array<std::uint32_t, 3u> too_many{2u, 2u, 2u};
  auto rejected =
      on(Target::cpu(2u), one)
          .join(
              MaxMatches{2u}, too_many, [](auto value) { return value; },
              [](auto value) { return value; },
              [](auto first, auto second) { return first + second; })
          .collect();
  if (rejected || rejected.error() != "compute_bounded_count_invalid") {
    return 2;
  }

  const std::array<std::uint32_t, 0u> empty{};
  auto empty_output =
      on(Target::cpu(2u), left)
          .join(
              MaxMatches{1u}, empty, [](auto value) { return value; },
              [](auto value) { return value; },
              [](auto first, auto second) { return first + second; })
          .collect();
  return empty_output && empty_output->empty() ? 0 : 3;
}

[[nodiscard]] int CheckGroupCapacity() {
  using namespace rund::compute;
  if constexpr (std::numeric_limits<std::size_t>::max() <=
                std::numeric_limits<std::uint32_t>::max()) {
    return 0;
  } else {
    constexpr std::size_t count =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) +
        1u;
    auto rejected = Target()
                        .map<std::uint64_t>("group-capacity", count,
                                            [](auto value) { return value; })
                        .group_by([](auto value) { return value; })
                        .aggregate([](auto group) {
                          return outputs(group.key(), group.count());
                        })
                        .compile();
    return !rejected && rejected.error() == "compute_group_capacity" ? 0 : 1;
  }
}

} // namespace

int RunComputeFlowPrimitivesContract() {
  if (const int result = CheckOutputs(); result != 0) {
    return 10 + result;
  }
  if (const int result = CheckIdentityProjection(); result != 0) {
    return 20 + result;
  }
  if (const int result = CheckComposition(); result != 0) {
    return 30 + result;
  }
  if (const int result = CheckBoundedPipe(); result != 0) {
    return 40 + result;
  }
  if (const int result = CheckBoundedGather(); result != 0) {
    return 45 + result;
  }
  if (const int result = CheckIndexedMap(); result != 0) {
    return 47 + result;
  }
  if (const int result = CheckScatterReduce(); result != 0) {
    return 48 + result;
  }
  if (const int result = CheckBoundedCollectiveRepeat(); result != 0) {
    return 52 + result;
  }
  if (const int result = CheckBoundedBoundaryConsumers(); result != 0) {
    return 60 + result;
  }
  if (const int result = CheckGroup(); result != 0) {
    return 70 + result;
  }
  if (const int result = CheckJoin(); result != 0) {
    return 80 + result;
  }
  if (const int result = CheckGroupCapacity(); result != 0) {
    return 90 + result;
  }
  return 0;
}
