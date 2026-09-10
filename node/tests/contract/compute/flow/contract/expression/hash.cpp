#include "local.hpp"

namespace rund_node_flow_contract::expression_detail {

int RunHashLane32(const rund::compute::Backend backend,
                  const std::array<FixedLane32, 3u> &input,
                  rund::compute::Stats &stats) {
  using T = FixedLane32;
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "hash-expression-surface", input.size(),
              [](auto value) {
                const auto half = fixed(FixedOp::Half, value);
                return record(
                    field<OpField<40>>(quantize<T>(hash(value))),
                    field<OpField<41>>(quantize<T>(hash(HashOp::Unit, value))),
                    field<OpField<42>>(quantize<T>(noise(value, half))),
                    field<OpField<43>>(quantize<T>(noise(value, half, half))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "hash expression compile failed backend=%u reason=%.*s\n",
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
  }(std::make_index_sequence<4u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

} // namespace rund_node_flow_contract::expression_detail
