#include <rund/compute.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>

int main() {
  constexpr std::size_t Count = 4u;
  auto device = rund::compute::open(rund::compute::Target::cpu(1u));
  if (!device) {
    return device.exit_code();
  }

  auto flow = rund::compute::on(*device).map<std::int32_t>(
      "affine", Count, [](auto value) { return value * 2 + 5; });
  auto program = std::move(flow).compile();
  if (!program) {
    return program.exit_code();
  }

  constexpr std::array<std::int32_t, Count> Initial{1, 2, 3, 4};
  constexpr std::array<std::int32_t, Count> Changed{10, 20, 30, 40};
  auto first = program->run(std::span<const std::int32_t>{Initial});
  auto second = program->run(std::span<const std::int32_t>{Changed});
  if (!first) {
    return first.exit_code();
  }
  if (!second) {
    return second.exit_code();
  }

  constexpr std::array<std::int32_t, Count> First{7, 9, 11, 13};
  constexpr std::array<std::int32_t, Count> Second{25, 45, 65, 85};
  return std::ranges::equal(*first, First) &&
                 std::ranges::equal(*second, Second)
             ? 0
             : 2;
}
