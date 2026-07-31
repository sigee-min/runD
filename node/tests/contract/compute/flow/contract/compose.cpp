#include "model.hpp"

#include <array>

namespace rund_node_flow_contract {

template <class T>
[[nodiscard]] bool
CheckUniformBroadcastSurface(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  const std::array<T, 4u> input{T{1}, T{2}, T{3}, T{4}};
  auto program = on(target)
                     .template input<T>(input.size())
                     .branch([](auto values) {
                       const auto maximum = values.reduce(Reduce::Max);
                       return values.combine("flow-uniform-read", maximum,
                                             [](auto value, auto scalar) {
                                               return value + scalar;
                                             });
                     })
                     .compile();
  if (!program || program->graph().authored_nodes != 2u ||
      program->graph().nodes.size() != 2u ||
      program->graph().resources.size() != 3u) {
    return false;
  }
  const graph::Node *map = nullptr;
  std::size_t gathers = 0u;
  for (const graph::Node &node : program->graph().nodes) {
    if (node.operation == graph::Operation::Map) {
      map = &node;
    }
    gathers += node.operation == graph::Operation::Gather ? 1u : 0u;
  }
  if (map == nullptr || gathers != 0u || map->elements != input.size() ||
      map->accesses.size() != 3u) {
    return false;
  }
  std::size_t scalar_reads = 0u;
  std::size_t sequence_reads = 0u;
  std::size_t sequence_writes = 0u;
  for (const graph::Access access : map->accesses) {
    if (access.mode == resource::AccessMode::Read &&
        access.element_count == 1u) {
      ++scalar_reads;
    } else if (access.mode == resource::AccessMode::Read &&
               access.element_count == input.size()) {
      ++sequence_reads;
    } else if (access.mode == resource::AccessMode::Write &&
               access.element_count == input.size()) {
      ++sequence_writes;
    }
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return false;
  }
  auto output = job->read();
  if (scalar_reads != 1u || sequence_reads != 1u || sequence_writes != 1u ||
      !output || *output != std::vector<T>{T{5}, T{6}, T{7}, T{8}} ||
      job->stats().backend != backend) {
    return false;
  }

  auto projected =
      on(target)
          .template input<T>(input.size())
          .branch([](auto values) {
            const auto maximum = values.reduce(Reduce::Max);
            return values.combine("flow-uniform-projection", maximum,
                                  [](auto, auto scalar) { return scalar; });
          })
          .compile();
  if (!projected || projected->graph().authored_nodes != 2u ||
      projected->graph().nodes.size() != 2u) {
    return false;
  }
  const graph::Node *projection = nullptr;
  for (const graph::Node &node : projected->graph().nodes) {
    if (node.operation == graph::Operation::Map) {
      projection = &node;
    } else if (node.operation == graph::Operation::Gather) {
      return false;
    }
  }
  if (projection == nullptr || projection->elements != input.size() ||
      projection->accesses.size() != 2u ||
      projection->accesses[0u].mode != resource::AccessMode::Read ||
      projection->accesses[0u].element_count != 1u ||
      projection->accesses[1u].mode != resource::AccessMode::Write ||
      projection->accesses[1u].element_count != input.size()) {
    return false;
  }
  auto projected_job = projected->resident(input);
  if (!projected_job || !projected_job->run()) {
    return false;
  }
  auto projected_output = projected_job->read();
  return projected_output &&
         *projected_output == std::vector<T>(input.size(), T{4}) &&
         projected_job->stats().backend == backend;
}

template <class T>
[[nodiscard]] bool CheckCompositionSurface(const rund::compute::Backend backend,
                                           FlowHash &reference) {
  using namespace rund::compute;
  const std::array<T, 4u> input{FlowDomainValue<T>(1), FlowDomainValue<T>(2),
                                FlowDomainValue<T>(3), FlowDomainValue<T>(4)};
  const std::array<T, 4u> side{FlowDomainValue<T>(4), FlowDomainValue<T>(3),
                               FlowDomainValue<T>(2), FlowDomainValue<T>(1)};
  const T one = FlowDomainValue<T>(1);
  auto program =
      FlowInput<T>(backend, "flow-composition", input.size())
          .combine(
              "flow-combine", side.size(),
              [](auto left, auto right) { return FlowStore<T>(left + right); })
          .pipe([one](auto values) {
            return values.template unroll<2u>([one](auto step) {
              return step.map("flow-repeat",
                              capture(
                                  [](auto value, auto constant) {
                                    return FlowStore<T>(value + constant);
                                  },
                                  one));
            });
          })
          .branch([](auto values) {
            const auto total = values.reduce(Reduce::Max);
            const auto scaled = values.combine(
                "flow-broadcast", total, [](auto value, auto maximum) {
                  return FlowStore<T>(value + maximum);
                });
            return outputs(record(values, total), scaled,
                           values.window({.op = Window::Max, .radius = 1u}));
          })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input, side);
  return job && CheckFlowParity(*job, backend, reference);
}

template <class T, class U>
[[nodiscard]] bool CheckTypedSurface(const rund::compute::Backend backend,
                                     FlowHash &reference) {
  using namespace rund::compute;
  const std::array<T, 4u> values{FlowDomainValue<T>(1), FlowDomainValue<T>(2),
                                 FlowDomainValue<T>(3), FlowDomainValue<T>(4)};
  const std::array<U, 4u> weights{FlowDomainValue<U>(4), FlowDomainValue<U>(3),
                                  FlowDomainValue<U>(2), FlowDomainValue<U>(1)};
  const auto target = on(rund::node::test_contract::target_for(backend, 2u));
  auto program =
      target.template input<T>(values.size())
          .template zip_input<U>(weights.size())
          .map("flow-typed-record",
               [](auto value, auto weight) {
                 return record(field<ValueField>(FlowStore<T>(value)),
                               field<WeightField>(FlowStore<U>(weight)),
                               field<ScoreField>(FlowStore<T>(value * weight)));
               })
          .branch([](auto rows) {
            const auto zipped =
                zip(rows.template get<ValueField>(),
                    rows.template get<WeightField>())
                    .map("flow-typed-zip", [](auto value, auto weight) {
                      return FlowStore<T>(value * weight);
                    });
            return outputs(rows.template get<ScoreField>(), zipped);
          })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(values, weights);
  return job && CheckFlowParity(*job, backend, reference);
}

[[nodiscard]] bool CheckComposition(const rund::compute::Backend backend,
                                    std::array<FlowHash, 6u> &references) {
  return CheckUniformBroadcastSurface<std::int32_t>(backend) &&
         CheckUniformBroadcastSurface<std::int64_t>(backend) &&
         CheckCompositionSurface<std::int32_t>(backend, references[0u]) &&
         CheckCompositionSurface<std::uint32_t>(backend, references[1u]) &&
         CheckCompositionSurface<std::int64_t>(backend, references[2u]) &&
         CheckCompositionSurface<std::uint64_t>(backend, references[3u]) &&
         CheckCompositionSurface<rund::compute::Fixed<1, 31>>(backend,
                                                              references[4u]) &&
         CheckCompositionSurface<rund::compute::Fixed<1, 63>>(backend,
                                                              references[5u]);
}

[[nodiscard]] bool CheckTyped(const rund::compute::Backend backend,
                              std::array<FlowHash, 3u> &references) {
  return CheckTypedSurface<std::int64_t, std::uint64_t>(backend,
                                                        references[0u]) &&
         CheckTypedSurface<rund::compute::Fixed<1, 31>,
                           rund::compute::Fixed<1, 31>>(backend,
                                                        references[1u]) &&
         CheckTypedSurface<rund::compute::Fixed<1, 63>,
                           rund::compute::Fixed<1, 63>>(backend,
                                                        references[2u]);
}

} // namespace rund_node_flow_contract
