#include "local.hpp"

namespace rund_node_flow_contract::expression_detail {

int RunCompositeLane32(const rund::compute::Backend backend,
                       const std::array<FixedLane32, 3u> &input,
                       rund::compute::Stats &stats) {
  using T = FixedLane32;
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "composite-expression-surface", input.size(),
              [](auto value) {
                const auto half = fixed(FixedOp::Half, value);
                return record(
                    field<OpField<20>>(
                        quantize<T>(dot(value, half, half, value))),
                    field<OpField<21>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            len(value, half))),
                    field<OpField<22>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            dist(value, half, half, value))),
                    field<OpField<23>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            mean(value, half))),
                    field<OpField<24>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            var(value, half))),
                    field<OpField<25>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            corr(value, half, half, value))),
                    field<OpField<26>>(quantize<T>(lerp(value, half, half))),
                    field<OpField<27>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(smoothstep(
                            fixed_zero(value), fixed_max(value), value))),
                    field<OpField<28>>(quantize<T>(
                        bandpass(value, fixed_zero(value), fixed_max(value)))),
                    field<OpField<29>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            ratio(value, half))),
                    field<OpField<30>>(quantize<T>(sum(value, half))),
                    field<OpField<31>>(quantize<T>(centered(value, half))),
                    field<OpField<32>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            zscore(value, half, half))),
                    field<OpField<33>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            angle(AngleOp::Cosine, value, half, half, value))),
                    field<OpField<34>>(quantize<T>(deadzone(value, half))),
                    field<OpField<35>>(quantize<T>(clip(value, half))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "composite expression compile failed backend=%u reason=%.*s\n",
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
