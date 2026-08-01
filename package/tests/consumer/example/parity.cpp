#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

int main() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};

  const auto execute = [&input](const rund::compute::Target target) {
    return rund::compute::on(target, input)
        .map("step", [](auto value) { return value * 2 + 1; })
        .collect();
  };

  const auto cpu = execute(rund::compute::Target::cpu());
  const auto metal = execute(rund::compute::Target::metal());
  const auto vulkan = execute(rund::compute::Target::vulkan());

  const auto report = [](const std::string_view target, const auto &result) {
    const std::string_view message = result.error();
    std::fprintf(stderr, "%.*s failed (code=%u): %.*s\n",
                 static_cast<int>(target.size()), target.data(),
                 static_cast<unsigned>(result.code()),
                 static_cast<int>(message.size()), message.data());
    return result.exit_code();
  };
  if (!cpu) {
    return report("cpu", cpu);
  }
  if (!metal) {
    return report("metal", metal);
  }
  if (!vulkan) {
    return report("vulkan", vulkan);
  }

  const auto same_bits = [](const auto &left, const auto &right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(),
                       left.size() * sizeof(std::int32_t)) == 0;
  };
  const std::vector<std::int32_t> expected{3, 5, 7, 9};
  if (*cpu != expected || !same_bits(*cpu, *metal) ||
      !same_bits(*metal, *vulkan)) {
    std::fputs("backend output mismatch\n", stderr);
    return 2;
  }

  std::printf("same bytes: cpu = metal = vulkan [%d, %d, %d, %d]\n", (*cpu)[0],
              (*cpu)[1], (*cpu)[2], (*cpu)[3]);
  return 0;
}
