#include "local.hpp"

#include <array>
#include <utility>
#include <vector>

namespace rund_node_flow_contract {

[[nodiscard]] int CheckBasic() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto output = rund::compute::on(rund::compute::Target::cpu(), input)
                    .map("twice", [](auto x) { return x * 2 + 5; })
                    .collect();
  if (!output || *output != std::vector<std::int32_t>{7, 9, 11, 13}) {
    return 1;
  }

  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<std::uint32_t> values(kCount);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(index % 7u);
  }
  auto prefix = rund::compute::on(rund::compute::Target::cpu(1u), values)
                    .map("adjust", [](auto x) { return x + 1; })
                    .scan(rund::compute::Scan::InclusiveSum)
                    .collect();
  if (!prefix || prefix->size() != values.size()) {
    return 2;
  }
  std::uint32_t sum = 0u;
  for (std::size_t index = 0; index < values.size(); ++index) {
    sum += values[index] + 1u;
    if ((*prefix)[index] != sum) {
      return 3;
    }
  }

  auto empty = rund::compute::on(rund::compute::Target::cpu(), input).collect();
  if (empty || empty.code() != rund::compute::Code::Invalid ||
      empty.error() != "compute_flow_empty") {
    return 4;
  }

  auto unnamed = rund::compute::on(rund::compute::Target::cpu(), input)
                     .map("", [](auto x) { return x + 1; })
                     .scan(rund::compute::Scan::InclusiveSum)
                     .collect();
  if (unnamed || unnamed.code() != rund::compute::Code::Invalid ||
      unnamed.error() != "compute_name_empty") {
    return 5;
  }

  std::array<std::int32_t, 4> borrowed_input{1, 2, 3, 4};
  auto borrowed =
      rund::compute::on(rund::compute::Target::cpu(1u), borrowed_input)
          .map("borrowed", [](auto x) { return x + 1; });
  borrowed_input[0] = 41;
  auto borrowed_output = std::move(borrowed).collect();
  if (!borrowed_output ||
      *borrowed_output != std::vector<std::int32_t>{42, 3, 4, 5}) {
    return 6;
  }

  ContiguousInput<std::int32_t, 4> custom{{1, 2, 3, 4}};
  auto custom_borrowed =
      rund::compute::on(rund::compute::Target::cpu(1u), custom)
          .map("custom-borrowed", [](auto x) { return x + 1; });
  custom.values[0] = 41;
  auto custom_output = std::move(custom_borrowed).collect();
  if (!custom_output ||
      *custom_output != std::vector<std::int32_t>{42, 3, 4, 5}) {
    return 14;
  }

  const std::int32_t c_array[]{1, 2, 3, 4};
  auto c_array_output = rund::compute::on(rund::compute::Target::cpu(), c_array)
                            .map("c-array", [](auto x) { return x * 2; })
                            .collect();
  if (!c_array_output ||
      *c_array_output != std::vector<std::int32_t>{2, 4, 6, 8}) {
    return 15;
  }

  const std::array<std::uint32_t, 4> parity_input{1, 2, 3, 4};
  const std::vector<std::uint32_t> parity_expected{2, 5, 9, 14};
  auto plan = rund::compute::on(rund::compute::Target::cpu(1u))
                  .map<std::uint32_t>("adjust", parity_input.size(),
                                      [](auto x) { return x + 1; })
                  .scan(rund::compute::Scan::InclusiveSum)
                  .compile();
  if (!plan) {
    return 7;
  }
  auto plan_output = plan->run(std::span<const std::uint32_t>{parity_input});
  if (!plan_output || *plan_output != parity_expected) {
    return 8;
  }

  auto many = rund::compute::on(rund::compute::Target::cpu(4u), parity_input)
                  .map("adjust", [](auto x) { return x + 1; })
                  .scan(rund::compute::Scan::InclusiveSum)
                  .collect();
  if (!many || *many != parity_expected) {
    return 9;
  }

  const std::array<std::int64_t, 3> wide_input{1, 2, 3};
  auto wide = rund::compute::on(rund::compute::Target::cpu(), wide_input)
                  .map("wide", [](auto x) { return x * 3; })
                  .collect();
  if (!wide || *wide != std::vector<std::int64_t>{3, 6, 9}) {
    return 11;
  }

  const std::array<std::uint64_t, 3> unsigned_wide_input{1, 2, 3};
  auto unsigned_wide =
      rund::compute::on(rund::compute::Target::cpu())
          .map<std::uint64_t>("unsigned-wide", unsigned_wide_input.size(),
                              [](auto x) { return x + 4; })
          .compile();
  if (!unsigned_wide) {
    return 12;
  }
  auto unsigned_wide_output =
      unsigned_wide->run(std::span<const std::uint64_t>{unsigned_wide_input});
  if (!unsigned_wide_output ||
      *unsigned_wide_output != std::vector<std::uint64_t>{5, 6, 7}) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_flow_contract
