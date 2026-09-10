#include "local.hpp"

namespace rund_node_flow_contract::expression_detail {

int RunFunctionalLane32(const rund::compute::Backend backend,
                        const std::array<FixedLane32, 3u> &input,
                        rund::compute::Stats &stats) {
  using T = FixedLane32;
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "functional-expression-surface", input.size(),
              [](auto value) {
                const auto zero = fixed_zero(value);
                const auto half = fixed(FixedOp::Half, value);
                const auto quarter = fixed(FixedOp::Quarter, value);
                const auto truth = gt(value, zero);
                const auto store = [](auto expression) {
                  return quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                  Approximation::Deterministic>(expression);
                };
                return record(
                    field<OpField<70>>(store(zero_if(
                        predicate_or(
                            predicate_not(eq(value, value)),
                            predicate_and(ge(value, quarter), le(value, half))),
                        keep_if(all(ne(value, zero), truth), neg(value))))),
                    field<OpField<71>>(store(
                        bit_xor(bit_and(value, half), bit_or(value, quarter)))),
                    field<OpField<72>>(store(
                        bandstop(clamp_range(saturate(value), quarter, half),
                                 quarter, half))),
                    field<OpField<73>>(store(select(
                        any(in_range(value, quarter, half),
                            out_range(value, zero, half)),
                        median(absdiff(value, half), midrange(value, quarter),
                               spread(value, half, quarter)),
                        step(quarter, value)))),
                    field<OpField<74>>(store(
                        diff(DifferenceOrder::Second, quarter, value, half))),
                    field<OpField<75>>(store(rms(value, half, quarter))),
                    field<OpField<76>>(store(
                        mix(poly(value, zero, half, quarter),
                            poly_deriv(value, half, quarter), half, half))),
                    field<OpField<77>>(store(
                        aff(Axis::X, half, quarter, quarter, value, half))),
                    field<OpField<78>>(
                        store(cross(value, half, half, quarter))),
                    field<OpField<79>>(
                        store(bary(Axis::X, value, quarter, zero, zero, half,
                                   zero, zero, half))),
                    field<OpField<80>>(store(add_sat(
                        unit(Axis::X, value, half),
                        add_sat(proj(Axis::Y, value, half, half, quarter),
                                reject(Axis::X, value, half, half, quarter))))),
                    field<OpField<81>>(
                        store(line(GeometryOp::Projection, Axis::X, value,
                                   quarter, zero, zero, half, half))),
                    field<OpField<82>>(store(activation(
                        ActivationOp::HardSwish,
                        activation(ActivationOp::LeakyRelu, value, quarter)))),
                    field<OpField<83>>(
                        store(add_sat(softsign(value), huber(value, quarter)))),
                    field<OpField<84>>(
                        store(add_sat(smootherstep(zero, half, value),
                                      window(WindowOp::Triangular, value)))),
                    field<OpField<85>>(store(add_sat(
                        remap(zero, half, quarter, half,
                              snap(value, half, quarter)),
                        add_sat(fade(value), add_sat(positive_part(value),
                                                     negative_part(value)))))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "functional expression compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  const bool reads_ok = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = job->template read<I>();
              return values && values->size() == input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<16u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

} // namespace rund_node_flow_contract::expression_detail
