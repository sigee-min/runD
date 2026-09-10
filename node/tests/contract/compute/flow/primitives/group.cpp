#include "local.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_test_flow_primitives {

[[nodiscard]] int CheckGroup() {
  using namespace rund::compute;
  const std::array<std::int32_t, 5u> input{-1, 2, -1, 0, -2};
  auto output =
      on(Target::cpu(2u), input)
          .branch([](auto values) {
            auto groups = values.group_by([](auto value) { return value; });
            auto aggregate = groups.aggregate([](auto group) {
              return record(group.key(), group.count(),
                            group.values().reduce(Reduce::Sum));
            });
            auto ordered =
                groups.values()
                    .map("group-double", [](auto value) { return value * 2; })
                    .scan(Scan::InclusiveSum)
                    .ordered();
            return outputs(aggregate, ordered);
          })
          .collect();
  if (!output) {
    std::fprintf(stderr, "flow group reason=%.*s\n",
                 static_cast<int>(output.error().size()),
                 output.error().data());
    return 1;
  }
  const auto &groups = std::get<0>(*output);
  return std::get<0>(groups) == std::vector<std::int32_t>{-2, -1, 0, 2} &&
                 std::get<1>(groups) ==
                     std::vector<std::uint32_t>{1u, 2u, 1u, 1u} &&
                 std::get<2>(groups) == std::vector<std::int32_t>{-2, -2, 0, 2} &&
                 std::get<1>(*output) ==
                     std::vector<std::int32_t>{-4, -2, -4, 0, 4}
             ? 0
             : 2;
}

[[nodiscard]] int CheckJoin() {
  using namespace rund::compute;
  const std::array<std::uint32_t, 3u> left{10u, 21u, 12u};
  const std::array<std::uint32_t, 4u> right{42u, 14u, 33u, 26u};
  auto output =
      on(Target::cpu(2u), left)
          .join(
              MaxMatches{3u}, right, [](auto value) { return value & 1u; },
              [](auto value) { return value & 1u; },
              [](auto first, auto second) {
                return record(field<Left>(first), field<Right>(second));
              })
          .collect();
  if (!output ||
      std::get<0>(*output) !=
          std::vector<std::uint32_t>{10u, 10u, 10u, 12u, 12u, 12u, 21u} ||
      std::get<1>(*output) !=
          std::vector<std::uint32_t>{42u, 14u, 26u, 42u, 14u, 26u, 33u}) {
    return 1;
  }

  const std::array<std::uint32_t, 1u> one{2u};
  const std::array<std::uint32_t, 3u> too_many{2u, 2u, 2u};
  auto rejected =
      on(Target::cpu(2u), one)
          .join(
              MaxMatches{2u}, too_many, [](auto value) { return value; },
              [](auto value) { return value; },
              [](auto first, auto second) { return first + second; })
          .collect();
  if (rejected || rejected.error() != "compute_bounded_count_invalid") {
    return 2;
  }

  const std::array<std::uint32_t, 0u> empty{};
  auto empty_output =
      on(Target::cpu(2u), left)
          .join(
              MaxMatches{1u}, empty, [](auto value) { return value; },
              [](auto value) { return value; },
              [](auto first, auto second) { return first + second; })
          .collect();
  return empty_output && empty_output->empty() ? 0 : 3;
}

[[nodiscard]] int CheckGroupCapacity() {
  using namespace rund::compute;
  if constexpr (std::numeric_limits<std::size_t>::max() <=
                std::numeric_limits<std::uint32_t>::max()) {
    return 0;
  } else {
    constexpr std::size_t count =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) +
        1u;
    auto rejected = Target()
                        .map<std::uint64_t>("group-capacity", count,
                                            [](auto value) { return value; })
                        .group_by([](auto value) { return value; })
                        .aggregate([](auto group) {
                          return outputs(group.key(), group.count());
                        })
                        .compile();
    return !rejected && rejected.error() == "compute_group_capacity" ? 0 : 1;
  }
}

} // namespace rund_node_test_flow_primitives
