#include "local.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace rund_node_test_flow_primitives {

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
  auto program = Target()
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

} // namespace rund_node_test_flow_primitives
