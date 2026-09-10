#include "local.hpp"

#include <rund/compute.hpp>

#include "../../target/selection.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

int CheckBatchReset(const rund::compute::Backend backend) {
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 60;
  }
  auto program =
      rund::compute::on(*device)
          .map<std::uint32_t>("batch-reset", 1u,
                              [](auto value) { return value; })
          .scatter(1u, {.count = 2u})
          .compile();
  constexpr std::array<std::uint32_t, 1u> first_value{7u};
  constexpr std::array<std::uint32_t, 1u> first_index{0u};
  constexpr std::array<std::uint32_t, 1u> second_value{8u};
  constexpr std::array<std::uint32_t, 1u> second_index{1u};
  if (!program) {
    return 61;
  }
  auto first = program->resident(first_value, first_index);
  auto second = program->resident(second_value, second_index);
  rund::compute::Batch batch{};
  if (!first || !second || !batch.add(*first) || !batch.add(*second) ||
      !batch.run()) {
    return 62;
  }
  auto first_output = first->read();
  auto second_output = second->read();
  constexpr std::array<std::uint32_t, 2u> expected_first{7u, 0u};
  constexpr std::array<std::uint32_t, 2u> expected_second{0u, 8u};
  if (!first_output || !second_output ||
      !std::equal(first_output->begin(), first_output->end(),
                  expected_first.begin(), expected_first.end()) ||
      !std::equal(second_output->begin(), second_output->end(),
                  expected_second.begin(), expected_second.end()) ||
      batch.stats().reset_bytes != 4u * sizeof(std::uint32_t) ||
      batch.stats().reset_commands != 2u ||
      first->stats().reset_bytes != 2u * sizeof(std::uint32_t) ||
      first->stats().reset_commands != 1u ||
      second->stats().reset_bytes != 2u * sizeof(std::uint32_t) ||
      second->stats().reset_commands != 1u) {
    return 63;
  }

  constexpr std::array<std::uint32_t, 1u> next_first_value{9u};
  constexpr std::array<std::uint32_t, 1u> next_first_index{1u};
  constexpr std::array<std::uint32_t, 1u> next_second_value{10u};
  constexpr std::array<std::uint32_t, 1u> next_second_index{0u};
  if (!first->write(next_first_value, next_first_index) ||
      !second->write(next_second_value, next_second_index) || !batch.run()) {
    return 64;
  }
  first_output = first->read();
  second_output = second->read();
  constexpr std::array<std::uint32_t, 2u> expected_next_first{0u, 9u};
  constexpr std::array<std::uint32_t, 2u> expected_next_second{10u, 0u};
  if (!first_output || !second_output ||
      !std::equal(first_output->begin(), first_output->end(),
                  expected_next_first.begin(), expected_next_first.end()) ||
      !std::equal(second_output->begin(), second_output->end(),
                  expected_next_second.begin(), expected_next_second.end()) ||
      batch.stats().reset_bytes != 4u * sizeof(std::uint32_t) ||
      batch.stats().reset_commands != 2u) {
    return 65;
  }
  return 0;
}
