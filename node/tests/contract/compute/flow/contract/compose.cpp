#include "model.hpp"

#include <array>

namespace rund_node_flow_contract {

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
  return CheckCompositionSurface<std::int32_t>(backend, references[0u]) &&
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
