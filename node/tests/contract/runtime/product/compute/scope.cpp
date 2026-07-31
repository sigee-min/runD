#include "../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

int RunRuntimeComputeScopeContract() {
  constexpr std::uint32_t kCoroutineFrameByteLimit = 8192u;
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  const std::array<std::int32_t, 4> sort_input{4, 1, 3, 2};
  auto program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-scope", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  TEST_ASSERT(program);
  auto job = program->resident(input);
  TEST_ASSERT(job);

  auto output_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-scope-outputs", input.size(),
                             [](auto value) { return value; })
          .branch([](auto source) {
            auto doubled =
                source.map("double", [](auto value) { return value * 2; });
            auto incremented =
                doubled.map("increment", [](auto value) { return value + 1; });
            auto mask = source.map("mask", [](auto value) {
              return rund::compute::select<std::uint32_t>(value > 2, 1u, 0u);
            });
            return rund::compute::outputs(doubled, incremented, mask);
          })
          .compile();
  TEST_ASSERT(output_program);
  auto output_job = output_program->resident(input);
  TEST_ASSERT(output_job);
  auto filter_program = rund::compute::on(rund::compute::Target::cpu(2u))
                            .map<std::int32_t>("compute-filter", input.size(),
                                               [](auto value) { return value; })
                            .filter([](auto value) { return value > 2; })
                            .compile();
  TEST_ASSERT(filter_program);
  auto filter_job = filter_program->resident(input);
  TEST_ASSERT(filter_job);
  auto count_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-filter-count", input.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .count()
          .compile();
  TEST_ASSERT(count_program);
  auto count_job = count_program->resident(input);
  TEST_ASSERT(count_job);
  auto reduce_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-filter-reduce", input.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .reduce(rund::compute::Reduce::Sum)
          .compile();
  TEST_ASSERT(reduce_program);
  auto reduce_job = reduce_program->resident(input);
  TEST_ASSERT(reduce_job);
  auto scan_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-filter-scan", input.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile();
  if (!scan_program) {
    std::fprintf(stderr, "compute-scope scan compile failed: %.*s\n",
                 static_cast<int>(scan_program.error().size()),
                 scan_program.error().data());
  }
  TEST_ASSERT(scan_program);
  auto scan_job = scan_program->resident(input);
  TEST_ASSERT(scan_job);
  auto sort_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-filter-sort", sort_input.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 1; })
          .sort()
          .compile();
  TEST_ASSERT(sort_program);
  auto sort_job = sort_program->resident(sort_input);
  TEST_ASSERT(sort_job);
  auto typed_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("compute-typed", input.size(),
                             [](auto value) {
                               return rund::compute::select<std::uint32_t>(
                                   value > 2, 1u, 0u);
                             })
          .compile();
  TEST_ASSERT(typed_program);
  auto typed_job = typed_program->resident(input);
  TEST_ASSERT(typed_job);

  bool completed = false;
  const rund::Session::Result report = rund::run(
      {.workers = 2u,
       .scheduler =
           {
               .coroutine_frame_bytes = kCoroutineFrameByteLimit,
           }},
      [&](rund::Session &session) {
        completed =
            static_cast<bool>(session.compute(*job).submit().wait()) &&
            static_cast<bool>(session.compute(*job).submit().wait()) &&
            static_cast<bool>(session.compute(*output_job).submit().wait()) &&
            static_cast<bool>(session.compute(*filter_job).submit().wait()) &&
            static_cast<bool>(session.compute(*count_job).submit().wait()) &&
            static_cast<bool>(session.compute(*reduce_job).submit().wait()) &&
            static_cast<bool>(session.compute(*scan_job).submit().wait()) &&
            static_cast<bool>(session.compute(*sort_job).submit().wait()) &&
            static_cast<bool>(session.compute(*typed_job).submit().wait());
      });
  TEST_ASSERT(report);
  TEST_ASSERT(completed);
  const auto node_memory = job->memory();
  TEST_ASSERT(node_memory.frame.current == 0u);
  TEST_ASSERT(node_memory.frame.peak != 0u);
  TEST_ASSERT(node_memory.frame.cumulative >= node_memory.frame.peak * 2u);
  TEST_ASSERT(node_memory.frame.reused != 0u);
  TEST_ASSERT(node_memory.frame.budget == kCoroutineFrameByteLimit);
  const auto output = job->read();
  TEST_ASSERT(output);
  TEST_ASSERT(*output == std::vector<std::int32_t>({7, 9, 11, 13}));
  const auto second = output_job->template read<1u>();
  TEST_ASSERT(second);
  TEST_ASSERT(*second == std::vector<std::int32_t>({3, 5, 7, 9}));
  const auto mask_output = output_job->template read<2u>();
  TEST_ASSERT(mask_output);
  TEST_ASSERT(*mask_output == std::vector<std::uint32_t>({0u, 0u, 1u, 1u}));
  const auto filtered = filter_job->read();
  TEST_ASSERT(filtered);
  TEST_ASSERT(*filtered == std::vector<std::int32_t>({3, 4}));
  const auto logical_count = count_job->read();
  TEST_ASSERT(logical_count);
  TEST_ASSERT(*logical_count == std::vector<std::uint32_t>({2u}));
  const auto reduced = reduce_job->read();
  TEST_ASSERT(reduced);
  TEST_ASSERT(*reduced == std::vector<std::int32_t>({7}));
  const auto scanned = scan_job->read();
  TEST_ASSERT(scanned);
  TEST_ASSERT(*scanned == std::vector<std::int32_t>({3, 7}));
  const auto sorted = sort_job->read();
  TEST_ASSERT(sorted);
  TEST_ASSERT(*sorted == std::vector<std::int32_t>({2, 3, 4}));
  const auto typed = typed_job->read();
  TEST_ASSERT(typed);
  TEST_ASSERT(*typed == std::vector<std::uint32_t>({0u, 0u, 1u, 1u}));
  return 0;
}
