#include <rund/compute.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

int main() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};

  const auto execute = [&input](const rund::compute::Device &device) {
    auto flow = rund::compute::on(device).map<std::int32_t>(
        "step", input.size(), [](auto value) { return value * 2 + 1; });
    auto program = std::move(flow).compile();
    if (!program) {
      return rund::compute::Result<std::vector<std::int32_t>>::fail(
          program.reason());
    }
    return program->run(std::span<const std::int32_t>{input});
  };

  const auto report = [](const std::string_view target, const auto &result) {
    const std::string_view message = result.error();
    std::fprintf(stderr, "%.*s failed (code=%u): %.*s\n",
                 static_cast<int>(target.size()), target.data(),
                 static_cast<unsigned>(result.code()),
                 static_cast<int>(message.size()), message.data());
    return result.exit_code();
  };
  auto cpu_device = rund::compute::open(rund::compute::Target::cpu());
  if (!cpu_device) {
    return report("cpu", cpu_device);
  }
  const auto cpu = execute(*cpu_device);
  if (!cpu) {
    return report("cpu", cpu);
  }

  const auto same_bits = [](const auto &left, const auto &right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(),
                       left.size() * sizeof(std::int32_t)) == 0;
  };
  const std::vector<std::int32_t> expected{3, 5, 7, 9};
  if (*cpu != expected) {
    std::fputs("cpu output mismatch\n", stderr);
    return 2;
  }

  std::size_t available_backend_count = 1u;
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
  constexpr bool require_accelerators = true;
#else
  constexpr bool require_accelerators = false;
#endif
  const auto compare = [&](const std::string_view name,
                           const rund::compute::Target target) {
    auto device = rund::compute::open(target);
    if (!device) {
      if (!require_accelerators &&
          device.reason() == rund::compute::Reason::AdapterUnavailable) {
        const std::string_view message = device.error();
        std::fprintf(stderr, "%.*s unavailable (code=%u): %.*s\n",
                     static_cast<int>(name.size()), name.data(),
                     static_cast<unsigned>(device.code()),
                     static_cast<int>(message.size()), message.data());
        return 0;
      }
      return report(name, device);
    }
    const auto result = execute(*device);
    if (!result) {
      return report(name, result);
    }
    if (!same_bits(*cpu, *result)) {
      std::fprintf(stderr, "%.*s output mismatch\n",
                   static_cast<int>(name.size()), name.data());
      return 2;
    }
    ++available_backend_count;
    return 0;
  };
  if (const int metal = compare("metal", rund::compute::Target::metal());
      metal != 0) {
    return metal;
  }
  if (const int vulkan = compare("vulkan", rund::compute::Target::vulkan());
      vulkan != 0) {
    return vulkan;
  }

  if (available_backend_count == 3u) {
    std::printf("same bytes: cpu = metal = vulkan [%d, %d, %d, %d]\n",
                (*cpu)[0], (*cpu)[1], (*cpu)[2], (*cpu)[3]);
  } else {
    std::printf("verified [%d, %d, %d, %d] on %zu native backend(s)\n",
                (*cpu)[0], (*cpu)[1], (*cpu)[2], (*cpu)[3],
                available_backend_count);
  }
  return 0;
}
