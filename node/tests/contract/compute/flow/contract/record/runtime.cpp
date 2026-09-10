#include "../../../../../../src/compute/cpu/state/program.hpp"
#include "../model.hpp"

#include "../../../../../../src/compute/flow/state.hpp"
#include "../../../../../../src/compute/program/state.hpp"

#include <array>
#include <vector>

namespace rund_node_flow_contract {

int CheckRecordRuntime(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{-2, 0, 3, 5};
  auto result = flow_on(backend, input)
                    .branch([](auto values) {
                      const auto rows =
                          values.map("typed-record", [](auto value) {
                            return record(field<ValueField>(value),
                                          field<WeightField>(mask(value > 0)),
                                          field<ScoreField>(value * 2));
                          });
                      const auto selected =
                          zip(rows.template get<ValueField>(),
                              rows.template get<WeightField>())
                              .map("typed-zip", [](auto value, auto weight) {
                                return value * weight;
                              });
                      return outputs(rows.template get<ScoreField>(), selected);
                    })
                    .collect();
  if (!result) {
    return 7;
  }
  if (std::get<0>(*result) != std::vector<std::int32_t>{-4, 0, 6, 10} ||
      std::get<1>(*result) != std::vector<std::uint32_t>{0, 0, 3, 5}) {
    return 8;
  }
  auto direct = flow_on(backend, input)
                    .map("direct-record",
                         [](auto value) {
                           return record(field<ValueField>(value),
                                         field<ScoreField>(value * 3));
                         })
                    .branch([](auto rows) {
                      return outputs(rows.template get<ScoreField>(),
                                     rows.template get<ValueField>());
                    })
                    .collect();
  if (!direct) {
    return 9;
  }
  if (std::get<0>(*direct) != std::vector<std::int32_t>{-6, 0, 9, 15} ||
      std::get<1>(*direct) != std::vector<std::int32_t>{-2, 0, 3, 5}) {
    return 10;
  }
  auto nested =
      flow_on(backend, input)
          .map("nested-record",
               [](auto value) {
                 return record(
                     field<ValueField>(value),
                     field<NestedField>(record(field<WeightField>(value * 4),
                                               field<ScoreField>(value * 5))));
               })
          .branch([](auto rows) {
            return outputs(
                rows.template get<NestedField>().template get<ScoreField>(),
                rows.template get<ValueField>());
          })
          .collect();
  if (!nested ||
      std::get<0>(*nested) != std::vector<std::int32_t>{-10, 0, 15, 25} ||
      std::get<1>(*nested) != std::vector<std::int32_t>{-2, 0, 3, 5}) {
    return 11;
  }
  auto selective =
      flow_on(backend)
          .input<std::int32_t>(input.size())
          .map("selective-record",
               [](auto value) {
                 return record(field<ValueField>(value * 2),
                               field<WeightField>(value * 3),
                               field<ScoreField>(value * 4));
               })
          .branch([](auto rows) { return rows.template get<ScoreField>(); })
          .compile();
  if (!selective) {
    return 12;
  }
  const graph::Info &selective_graph = selective->graph();
  if (selective_graph.nodes.size() != 1u ||
      selective_graph.nodes.front().operation != graph::Operation::Map ||
      selective_graph.nodes.front().accesses.size() != 2u ||
      selective_graph.nodes.front().accesses[0u].mode !=
          resource::AccessMode::Read ||
      selective_graph.nodes.front().accesses[1u].mode !=
          resource::AccessMode::Write) {
    return 13;
  }
  if (backend == Backend::Cpu) {
    const auto &state = detail::FlowAccess::state(*selective);
    if (state == nullptr || state->cpu_graph == nullptr ||
        state->cpu_graph->maps.size() != 1u ||
        state->cpu_graph->maps.front() == nullptr) {
      return 13;
    }
    const auto &cpu_map = *state->cpu_graph->maps.front();
    const auto &prepared = cpu_map.dispatch.prepared;
    if (cpu_map.map.output_buffer_count != 1u ||
        prepared.instructions.empty() || prepared.value_slot_count == 0u ||
        prepared.value_slot_count > prepared.instructions.size() ||
        prepared.once_count > prepared.instructions.size()) {
      return 13;
    }
  }
  auto selective_job = selective->resident(input);
  if (!selective_job || !selective_job->run()) {
    return 14;
  }
  auto selective_output = selective_job->read();
  const Stats selective_stats = selective_job->stats();
  if (!selective_output ||
      *selective_output != std::vector<std::int32_t>{-8, 0, 12, 20} ||
      selective_stats.graph_hash == 0u || selective_stats.output_hash == 0u) {
    return 15;
  }
  auto retained_internal =
      flow_on(backend)
          .input<std::int32_t>(input.size())
          .map("retained-map", [](auto value) { return value * 2; })
          .scan(Scan::InclusiveSum)
          .compile();
  if (!retained_internal) {
    return 40;
  }
  auto retained_job = retained_internal->resident(input);
  if (!retained_job) {
    return 42;
  }
  if (!retained_job->run()) {
    return 43;
  }
  const auto retained_output = retained_job->read();
  const Stats retained_stats = retained_job->stats();
  if (!retained_output ||
      *retained_output != std::vector<std::int32_t>{-4, -4, 2, 12} ||
      retained_stats.graph_hash == 0u || retained_stats.output_hash == 0u) {
    return 44;
  }
  return 0;
}

} // namespace rund_node_flow_contract
