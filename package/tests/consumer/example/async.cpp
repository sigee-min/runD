#include <rund/compute.hpp>
#include <rund/compute/async.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

int main() {
  constexpr std::size_t Count = 4u;
  auto device = rund::compute::open(
      rund::compute::Target::cpu(1u),
      rund::compute::Compile{.workers = 1u, .capacity = 2u});
  if (!device) {
    return device.exit_code();
  }

  auto pending =
      rund::compute::on(*device)
          .map<std::int32_t>("async-adjust", Count,
                             [](auto value) { return value * 2 + 5; })
          .compile_async();
  if (!pending) {
    return pending.exit_code();
  }

  auto program = pending->get();
  if (!program) {
    return program.exit_code();
  }

  constexpr std::array<std::int32_t, Count> input{1, 2, 3, 4};
  auto output = program->run(std::span<const std::int32_t>{input});
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{7, 9, 11, 13} ? 0 : 2;
}
