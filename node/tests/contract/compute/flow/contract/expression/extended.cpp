#include "local.hpp"

namespace rund_node_flow_contract::expression_detail {

int RunExtendedLane32(const rund::compute::Backend backend,
                      const std::array<FixedLane32, 3u> &input,
                      rund::compute::Stats &stats) {
  using T = FixedLane32;
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "extended-expression-surface", input.size(),
              [](auto value) {
                const auto half = fixed(FixedOp::Half, value);
                const auto quarter = fixed(FixedOp::Quarter, value);
                const auto store = [](auto expression) {
                  return quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                  Approximation::Deterministic>(expression);
                };
                return record(
                    field<OpField<50>>(store(conv(
                        value, half, quarter, value, half, quarter, value, half,
                        quarter, value, half, quarter, value, half))),
                    field<OpField<51>>(store(value)),
                    field<OpField<52>>(store(standardized(
                        StandardizedOp::Cubic, value, half, quarter))),
                    field<OpField<53>>(store(standardized(
                        StandardizedOp::Quartic, value, half, quarter))),
                    field<OpField<54>>(store(
                        mean(StandardizedOp::Cubic, value, half, quarter))),
                    field<OpField<55>>(store(cov(value, half, quarter, value,
                                                 half, quarter, value, half))),
                    field<OpField<56>>(store(corr(value, half, quarter, value,
                                                  half, quarter, value, half))),
                    field<OpField<57>>(
                        store(proportion(Axis::Z, value, half, quarter))),
                    field<OpField<58>>(store(reflect(
                        Axis::Z, value, half, quarter, half, quarter, value))),
                    field<OpField<59>>(store(
                        plane(GeometryOp::Parameter, value, half, quarter, half,
                              quarter, value, value, half, quarter))),
                    field<OpField<60>>(store(
                        plane(GeometryOp::Distance, value, half, quarter, half,
                              quarter, value, value, half, quarter))),
                    field<OpField<61>>(store(plane(
                        GeometryOp::Projection, Axis::Y, value, half, quarter,
                        half, quarter, value, value, half, quarter))),
                    field<OpField<62>>(
                        store(weighted_mean(value, half, quarter, value, half,
                                            quarter, half, quarter))),
                    field<OpField<63>>(
                        store(bezier(value, half, quarter, half))),
                    field<OpField<64>>(
                        store(bezier(value, half, quarter, value, half))),
                    field<OpField<65>>(
                        store(mat(MatOp::Transpose, Axis::Z, value, half,
                                  quarter, half, quarter, value))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "extended expression compile failed backend=%u reason=%.*s\n",
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
