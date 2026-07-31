#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

int main() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto output = rund::compute::on(rund::compute::Target::cpu(), input)
                    .map("twice", [](auto value) { return value * 2 + 5; })
                    .collect();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{7, 9, 11, 13} ? 0 : 2;
}
