#include "../../../../../../src/compute/cpu/state/program.hpp"
#include "../model.hpp"

#include "../../../../../../src/compute/flow/state.hpp"
#include "../../../../../../src/compute/program/state.hpp"

#include <array>
#include <vector>

namespace rund_node_flow_contract {

int CheckRecordSchema(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{-2, 0, 3, 5};
  const std::array<std::uint32_t, 4u> weights{3u, 4u, 5u, 6u};
  auto selective_inputs =
      flow_on(backend)
          .input<std::int32_t>(input.size())
          .zip_input<std::uint32_t>(weights.size())
          .map("selective-input-record",
               [](auto value, auto weight) {
                 return record(field<ValueField>(value * 2),
                               field<WeightField>(weight * 3u));
               })
          .branch([](auto rows) { return rows.template get<ValueField>(); })
          .compile();
  if (!selective_inputs) {
    return 30;
  }
  const graph::Info &selective_inputs_graph = selective_inputs->graph();
  if (selective_inputs_graph.nodes.size() != 1u ||
      selective_inputs_graph.nodes.front().operation != graph::Operation::Map ||
      selective_inputs_graph.nodes.front().accesses.size() != 2u ||
      selective_inputs_graph.nodes.front().accesses[0u].mode !=
          resource::AccessMode::Read ||
      selective_inputs_graph.nodes.front().accesses[1u].mode !=
          resource::AccessMode::Write) {
    return 31;
  }
  if (backend == Backend::Cpu) {
    const auto &state = detail::FlowAccess::state(*selective_inputs);
    if (state == nullptr || state->cpu_graph == nullptr ||
        state->cpu_graph->maps.size() != 1u ||
        state->cpu_graph->maps.front() == nullptr ||
        state->cpu_graph->maps.front()->map.input_buffer_count != 1u ||
        state->cpu_graph->maps.front()->map.output_buffer_count != 1u) {
      return 31;
    }
  }
  auto selective_inputs_job = selective_inputs->resident(input, weights);
  if (!selective_inputs_job || !selective_inputs_job->run()) {
    return 32;
  }
  auto selective_inputs_output = selective_inputs_job->read();
  if (!selective_inputs_output ||
      *selective_inputs_output != std::vector<std::int32_t>{-4, 0, 6, 10}) {
    return 33;
  }
  auto bounded_record = flow_on(backend, input)
                            .filter([](auto value) { return value > 0; })
                            .map("bounded-record",
                                 [](auto value) {
                                   return record(field<ValueField>(value * 2),
                                                 field<ScoreField>(value * 3));
                                 })
                            .collect();
  if (!bounded_record ||
      std::get<0>(*bounded_record) != std::vector<std::int32_t>{6, 10} ||
      std::get<1>(*bounded_record) != std::vector<std::int32_t>{9, 15}) {
    return 16;
  }
  auto bounded_plan = flow_on(backend)
                          .map<std::int32_t>("bounded-input", input.size(),
                                             [](auto value) { return value; })
                          .filter([](auto value) { return value > 0; })
                          .map("bounded-record-plan",
                               [](auto value) {
                                 return record(field<ValueField>(value * 4),
                                               field<ScoreField>(value * 5));
                               })
                          .compile();
  if (!bounded_plan) {
    return 17;
  }
  auto bounded_job = bounded_plan->resident(input);
  if (!bounded_job || !bounded_job->run()) {
    return 18;
  }
  auto bounded_values = bounded_job->template read<0u>();
  auto bounded_scores = bounded_job->template read<1u>();
  if (!bounded_values || !bounded_scores ||
      *bounded_values != std::vector<std::int32_t>{12, 20} ||
      *bounded_scores != std::vector<std::int32_t>{15, 25}) {
    return 19;
  }
  auto alternate_plan =
      flow_on(backend)
          .map<std::int32_t>("alternate-input", input.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 0; })
          .map("alternate-record",
               [](auto value) {
                 return record(field<AlternateValueField>(value * 4),
                               field<AlternateScoreField>(value * 5));
               })
          .compile();
  if (!alternate_plan) {
    return 20;
  }
  auto alternate_job = alternate_plan->resident(input);
  if (!alternate_job || !alternate_job->run()) {
    return 21;
  }
  return alternate_job->stats().graph_hash == bounded_job->stats().graph_hash
             ? 0
             : 22;
}

} // namespace rund_node_flow_contract
