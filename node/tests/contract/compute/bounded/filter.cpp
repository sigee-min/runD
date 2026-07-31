#include "model.hpp"

#include "../../target/selection.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rund_node_bounded_contract {

template <class T> bool CheckFilterType(const rund::compute::Target target) {
  const std::array<T, 5> input{Small<T>(5u), Small<T>(1u), Small<T>(4u),
                               Small<T>(2u), Small<T>(3u)};
  auto result = rund::compute::on(target, input)
                    .filter([](auto value) { return value > Small<T>(2u); })
                    .collect();
  auto refiltered = rund::compute::on(target, input)
                        .filter([](auto value) { return value > Small<T>(1u); })
                        .filter([](auto value) { return value > Small<T>(2u); })
                        .collect();
  const std::vector<T> expected{Small<T>(5u), Small<T>(4u), Small<T>(3u)};
  const bool ok =
      result && refiltered && *result == expected && *refiltered == expected;
  if (!ok) {
    const auto first = result ? std::string_view{} : result.error();
    const auto second = refiltered ? std::string_view{} : refiltered.error();
    std::fprintf(stderr,
                 "filter type bytes=%zu signed=%u first=%.*s second=%.*s\n",
                 sizeof(T), std::is_signed_v<T> ? 1u : 0u,
                 static_cast<int>(first.size()), first.data(),
                 static_cast<int>(second.size()), second.data());
  }
  return ok;
}

bool CheckStableFilter(const rund::compute::Target target) {
  std::array<std::int32_t, 1025u> input{};
  std::vector<std::int32_t> expected;
  expected.reserve(684u);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const auto magnitude = static_cast<std::int32_t>(100000u - index);
    input[index] = index % 3u == 1u ? -magnitude : magnitude;
    if (input[index] > 0) {
      expected.push_back(input[index]);
    }
  }
  auto result = rund::compute::on(target, input)
                    .filter([](auto value) { return value > 0; })
                    .collect();
  return result && *result == expected;
}

bool CheckResidentCountLineage(const rund::compute::Target target) {
  using namespace rund::compute;
  const std::array<std::int64_t, 5u> input{5, 1, 4, 2, 3};
  const std::array<std::uint32_t, 4u> right{0u, 1u, 2u, 3u};
  auto program =
      on(target)
          .template input<std::int64_t>(input.size())
          .template zip_input<std::uint32_t>(right.size())
          .branch([](auto values, auto matches) {
            auto order =
                values.filter([](auto value) { return value > 1; }).argsort();
            return outputs(
                order.filter([](auto index) { return index > 1u; }),
                order.scan(Scan::InclusiveSum), order.reduce(Reduce::Sum),
                order.sort(),
                order.window({.op = Window::Sum,
                              .radius = 1u,
                              .edge = WindowEdge::Clamp}),
                order.expand(
                    MaxItems{2u}, [](auto index) { return index & 1u; },
                    [](auto index, auto local) { return index + local; }),
                order.join(
                    MaxMatches{1u}, matches, [](auto value) { return value; },
                    [](auto value) { return value; },
                    [](auto left, auto matched) { return left + matched; }));
          })
          .compile();
  auto grouped_program =
      on(target)
          .template input<std::int64_t>(input.size())
          .branch([](auto values) {
            auto order =
                values.filter([](auto value) { return value > 1; }).argsort();
            return order.group_by([](auto index) { return index & 1u; })
                .aggregate([](auto group) {
                  return outputs(group.key(), group.count(),
                                 group.values().reduce(Reduce::Sum));
                });
          })
          .compile();
  if (!program || !grouped_program) {
    const auto reason = !program ? program.error() : grouped_program.error();
    std::fprintf(stderr, "bounded lineage compile reason=%.*s\n",
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  auto job = program->resident(input, right);
  auto grouped_job = grouped_program->resident(input);
  if (!job || !grouped_job) {
    const auto reason = !job ? job.error() : grouped_job.error();
    std::fprintf(stderr, "bounded lineage resident reason=%.*s\n",
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  const Status run = job->run();
  const Status grouped_run = grouped_job->run();
  if (!run || !grouped_run) {
    const auto reason = !run ? run.error() : grouped_run.error();
    std::fprintf(stderr,
                 "bounded lineage run reason=%.*s primary=%u grouped=%u\n",
                 static_cast<int>(reason.size()), reason.data(),
                 static_cast<unsigned>(run.reason()),
                 static_cast<unsigned>(grouped_run.reason()));
    return false;
  }
  auto result = job->read_all();
  auto grouped_result = grouped_job->read_all();
  if (!result || !grouped_result) {
    const auto reason = !result ? result.error() : grouped_result.error();
    std::fprintf(stderr, "bounded lineage read reason=%.*s\n",
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  const Stats stats = job->stats();
  const Stats grouped_stats = grouped_job->stats();
  const std::array matches{
      std::get<0>(*result) == std::vector<std::uint32_t>{2u, 3u},
      std::get<1>(*result) == std::vector<std::uint32_t>{2u, 5u, 6u, 6u},
      std::get<2>(*result) == 6u,
      std::get<3>(*result) == std::vector<std::uint32_t>{0u, 1u, 2u, 3u},
      std::get<4>(*result) == std::vector<std::uint32_t>{7u, 6u, 4u, 1u},
      std::get<5>(*result) == std::vector<std::uint32_t>{3u, 1u},
      std::get<6>(*result) == std::vector<std::uint32_t>{0u, 2u, 4u, 6u},
      std::get<0>(*grouped_result) == std::vector<std::uint32_t>{0u, 1u},
      std::get<1>(*grouped_result) == std::vector<std::uint32_t>{2u, 2u},
      std::get<2>(*grouped_result) == std::vector<std::uint32_t>{2u, 4u},
      stats.graph_hash != 0u && stats.output_hash != 0u,
      grouped_stats.graph_hash != 0u && grouped_stats.output_hash != 0u};
  const bool same = std::all_of(matches.begin(), matches.end(),
                                [](const bool value) { return value; });
  if (!same) {
    std::fprintf(stderr,
                 "bounded lineage mismatch rows="
                 "%d%d%d%d%d%d%d/%d%d%d/%d%d\n",
                 matches[0], matches[1], matches[2], matches[3], matches[4],
                 matches[5], matches[6], matches[7], matches[8], matches[9],
                 matches[10], matches[11]);
    std::fprintf(stderr, "bounded lineage expand size=%zu values=",
                 std::get<5>(*result).size());
    for (const std::uint32_t value : std::get<5>(*result)) {
      std::fprintf(stderr, "%u,", value);
    }
    std::fprintf(stderr, " join size=%zu values=", std::get<6>(*result).size());
    for (const std::uint32_t value : std::get<6>(*result)) {
      std::fprintf(stderr, "%u,", value);
    }
    std::fputc('\n', stderr);
  }
  return same;
}

int CheckFilterBackend(const rund::compute::Target target,
                       rund::compute::Stats *const evidence) {
  using namespace rund::compute;
  if (!CheckFilterType<std::int32_t>(target) ||
      !CheckFilterType<std::uint32_t>(target) ||
      !CheckFilterType<std::int64_t>(target) ||
      !CheckFilterType<std::uint64_t>(target) ||
      !CheckFilterType<Fixed<1, 31>>(target) ||
      !CheckFilterType<Fixed<1, 63>>(target)) {
    return 1;
  }
  if (!CheckStableFilter(target)) {
    return 16;
  }
  const std::array<std::int32_t, 4> signed32{-1, 0, 2, 3};
  auto mask32 = on(target, signed32)
                    .map("typed-mask32",
                         [](auto value) {
                           return select<std::uint32_t>(value > 0, 1u, 0u);
                         })
                    .collect();
  if (!mask32 || *mask32 != std::vector<std::uint32_t>{0u, 0u, 1u, 1u}) {
    return 6;
  }
  const std::array<std::int64_t, 4> signed64{-1, 0, 2, 3};
  auto mask64 = on(target, signed64)
                    .map("typed-mask64",
                         [](auto value) {
                           return select<std::uint64_t>(value > 0, 1u, 0u);
                         })
                    .collect();
  if (!mask64 || *mask64 != std::vector<std::uint64_t>{0u, 0u, 1u, 1u}) {
    return 7;
  }
  const std::array<std::int64_t, 4> input{1, 2, 3, 4};
  auto count = on(target, input)
                   .filter([](auto value) { return value > 2; })
                   .count()
                   .collect();
  if (!count || *count != std::vector<std::uint64_t>{2u}) {
    return 8;
  }
  if (!CheckResidentCountLineage(target)) {
    return 17;
  }
  auto bounded_sum = on(target, input)
                         .filter([](auto value) { return value > 2; })
                         .reduce(Reduce::Sum)
                         .collect();
  if (!bounded_sum || *bounded_sum != std::vector<std::int64_t>{7}) {
    return 13;
  }
  auto empty_sum = on(target, input)
                       .filter([](auto value) { return value > 9; })
                       .reduce(Reduce::Sum)
                       .collect();
  if (!empty_sum || *empty_sum != std::vector<std::int64_t>{0}) {
    return 14;
  }
  auto empty_min = on(target, input)
                       .filter([](auto value) { return value > 9; })
                       .reduce(Reduce::Min)
                       .collect();
  if (empty_min || empty_min.error() != "compute_reduce_count_zero") {
    return 15;
  }
  auto program =
      on(target)
          .template map<std::int64_t>("filter-resident", input.size(),
                                      [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .map("filter-resident-double", [](auto value) { return value * 2; })
          .compile();
  if (!program) {
    return 2;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 3;
  }
  const Stats stats = job->stats();
  const Status warm_run = job->run();
  if (stats.pipeline_compiles != 0u || stats.buffer_allocations != 0u ||
      stats.download_events != 0u || !warm_run) {
    std::fprintf(stderr,
                 "bounded warm pipeline_compiles=%llu buffer_allocations=%llu "
                 "download_events=%llu reason=%.*s\n",
                 static_cast<unsigned long long>(stats.pipeline_compiles),
                 static_cast<unsigned long long>(stats.buffer_allocations),
                 static_cast<unsigned long long>(stats.download_events),
                 static_cast<int>(warm_run.error().size()),
                 warm_run.error().data());
    return 4;
  }
  auto result = job->read();
  if (!result || *result != std::vector<std::int64_t>{6, 8}) {
    return 5;
  }
  const Stats observed = job->stats();
  auto count_program =
      on(target)
          .template map<std::int64_t>("filter-count-resident", input.size(),
                                      [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .count()
          .compile();
  if (!count_program) {
    return 9;
  }
  auto count_job = count_program->resident(input);
  if (!count_job || !count_job->run()) {
    return 10;
  }
  const Stats count_stats = count_job->stats();
  if (count_stats.download_events != 0u ||
      count_stats.dispatches >= stats.dispatches) {
    return 11;
  }
  auto resident_count = count_job->read();
  if (!resident_count || *resident_count != std::vector<std::uint64_t>{2u}) {
    return 12;
  }
  if (evidence != nullptr) {
    *evidence = observed;
  }
  return 0;
}

int CheckAcceleratorFilterLaws(const rund::compute::Backend backend) {
  using F32 = rund::compute::Fixed<1, 31>;
  using F64 = rund::compute::Fixed<1, 63>;
  const auto target = rund::node::test_contract::target_for(backend);
  if (!CheckFilterType<F32>(target) || !CheckFilterType<F64>(target)) {
    return 1;
  }
  if (!CheckStableFilter(target)) {
    return 2;
  }
  return CheckResidentCountLineage(target) ? 0 : 3;
}

} // namespace rund_node_bounded_contract
