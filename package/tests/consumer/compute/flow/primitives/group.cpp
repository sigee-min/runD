#include "model.hpp"

namespace package_compute {

int Group() {
  auto expanded = rund::compute::on(rund::compute::Target::cpu(), values)
                      .expand(
                          rund::compute::MaxItems{2u},
                          [](auto value) { return value & 1u; },
                          [](auto value, auto local) { return value + local; })
                      .collect();
  if (!expanded) {
    return expanded.exit_code();
  }
  if (*expanded != std::vector<std::uint32_t>{3u, 1u}) {
    return FlowMismatch(__LINE__);
  }

  auto grouped = rund::compute::on(rund::compute::Target::cpu(), values)
                     .group_by([](auto value) { return value & 1u; })
                     .aggregate([](auto group) {
                       return rund::compute::record(
                           group.key(), group.count(),
                           group.values().reduce(rund::compute::Reduce::Sum));
                     })
                     .collect();
  if (!grouped) {
    return grouped.exit_code();
  }
  if (std::get<0>(std::get<0>(*grouped)) !=
          std::vector<std::uint32_t>{0u, 1u} ||
      std::get<1>(std::get<0>(*grouped)) !=
          std::vector<std::uint32_t>{2u, 2u} ||
      std::get<2>(std::get<0>(*grouped)) !=
          std::vector<std::uint32_t>{6u, 4u}) {
    return FlowMismatch(__LINE__);
  }

  const std::array<std::uint64_t, 5u> wide_values{3u, 1u, 3u, 2u, 1u};
  auto wide_grouped =
      rund::compute::on(rund::compute::Target::cpu(), wide_values)
          .group_by([](auto value) { return value; })
          .aggregate([](auto group) {
            return rund::compute::record(
                group.key(), group.count(),
                group.values().reduce(rund::compute::Reduce::Sum));
          })
          .collect();
  if (!wide_grouped) {
    return wide_grouped.exit_code();
  }
  if (std::get<0>(std::get<0>(*wide_grouped)) !=
          std::vector<std::uint64_t>{1u, 2u, 3u} ||
      std::get<1>(std::get<0>(*wide_grouped)) !=
          std::vector<std::uint32_t>{2u, 1u, 2u} ||
      std::get<2>(std::get<0>(*wide_grouped)) !=
          std::vector<std::uint64_t>{2u, 2u, 6u}) {
    return FlowMismatch(__LINE__);
  }

  auto joined = rund::compute::on(rund::compute::Target::cpu(), values)
                    .join(
                        rund::compute::MaxMatches{2u}, combine_side,
                        [](auto value) { return value & 1u; },
                        [](auto value) { return value & 1u; },
                        [](auto left, auto right) { return left + right; })
                    .collect();
  if (!joined) {
    return joined.exit_code();
  }
  if (*joined != std::vector<std::uint32_t>{6u, 8u, 4u, 6u, 4u, 6u, 2u, 4u}) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
