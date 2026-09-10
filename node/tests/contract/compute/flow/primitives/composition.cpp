#include "local.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace rund_node_test_flow_primitives {

[[nodiscard]] int CheckComposition() {
  using namespace rund::compute;
  const std::array<std::uint64_t, 4u> input{3u, 1u, 4u, 2u};
  const std::array<std::uint64_t, 4u> side{4u, 3u, 2u, 1u};

  auto result =
      on(rund::compute::Target::cpu(2u), input)
          .combine("pair-sum", side,
                   [](auto left, auto right) { return left + right; })
          .pipe([](auto values) {
            return values.map("double", [](auto value) { return value * 2u; });
          })
          .branch([](auto values) {
            auto repeated = values.template unroll<2u>([](auto stage) {
              return stage.map("increment",
                               [](auto value) { return value + 1u; });
            });
            auto total = repeated.reduce(Reduce::Sum);
            auto fields = record(field<Values>(repeated), field<Sum>(total));
            auto adjusted = fields.template get<Values>().combine(
                "add-total", fields.template get<Sum>(),
                [](auto value, auto sum) { return value + sum; });
            return outputs(fields, adjusted, values.indices());
          })
          .collect();
  if (!result) {
    std::fprintf(stderr, "flow composition reason=%.*s\n",
                 static_cast<int>(result.error().size()),
                 result.error().data());
    return 1;
  }
  const auto &fields = std::get<0>(*result);
  return std::get<0>(fields) == std::vector<std::uint64_t>{16u, 10u, 14u, 8u} &&
                 std::get<1>(fields) == 48u &&
                 std::get<1>(*result) ==
                     std::vector<std::uint64_t>{64u, 58u, 62u, 56u} &&
                 std::get<2>(*result) ==
                     std::vector<std::uint64_t>{0u, 1u, 2u, 3u}
             ? 0
             : 2;
}

[[nodiscard]] int CheckBoundedPipe() {
  using namespace rund::compute;
  const std::array<std::uint32_t, 4u> input{3u, 1u, 4u, 2u};
  const auto normalize = [](auto values) {
    auto total = values.reduce(Reduce::Sum);
    return values.combine("normalize", total, [](auto value, auto sum) {
      return value * 90u / sum;
    });
  };
  auto output = on(Target::cpu(2u), input)
                    .filter([](auto value) { return value > 1u; })
                    .pipe(normalize)
                    .collect();
  return output && *output == std::vector<std::uint32_t>{30u, 40u, 20u} ? 0 : 1;
}

} // namespace rund_node_test_flow_primitives
