#include "local.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <span>
#include <thread>

namespace rund_node_test_compute_reuse {

[[nodiscard]] bool CheckProgramConcurrency(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t count = 32u;
  std::array<std::int32_t, count> first{};
  std::array<std::int32_t, count> second{};
  std::array<std::int32_t, count> first_expected{};
  std::array<std::int32_t, count> second_expected{};
  for (std::size_t index = 0u; index < count; ++index) {
    first[index] = static_cast<std::int32_t>(index);
    second[index] = -static_cast<std::int32_t>(index);
    first_expected[index] = first[index] * 3 + 1;
    second_expected[index] = second[index] * 3 + 1;
  }

  auto compiled =
      on(device)
          .map<std::int32_t>("program-concurrency", count,
                             [](auto value) { return value * 3 + 1; })
          .compile();
  if (!compiled) {
    return false;
  }
  auto program = std::move(compiled).value();
  std::atomic<bool> failed{false};
  const auto run = [&](const std::array<std::int32_t, count> &input,
                       const std::array<std::int32_t, count> &expected) {
    for (std::size_t iteration = 0u; iteration < 128u; ++iteration) {
      auto output = program.run(std::span<const std::int32_t>{input});
      if (!output || output->size() != expected.size() ||
          !std::equal(output->begin(), output->end(), expected.begin())) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  };
  std::thread first_run{run, std::cref(first), std::cref(first_expected)};
  std::thread second_run{run, std::cref(second), std::cref(second_expected)};
  std::thread observer{[&] {
    for (std::size_t iteration = 0u; iteration < 128u; ++iteration) {
      if (program.memory().scope != MemoryScope::Program) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  }};
  first_run.join();
  second_run.join();
  observer.join();
  return !failed.load(std::memory_order_relaxed);
}

[[nodiscard]] bool CheckProgramLifetime(rund::compute::Device &device) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};

  const MemoryStats before = device.memory();
  {
    auto compiled =
        on(device)
            .map<std::int32_t>("program-cache-lifetime", input.size(),
                               [](auto value) { return value + 1; })
            .compile();
    if (!compiled || !compiled->run(input)) {
      return false;
    }
  }
  if (!SameMemory(before, device.memory())) {
    std::fprintf(stderr, "program convenience cache outlived its owner\n");
    return false;
  }

  auto source = device.upload(std::span<const std::int32_t>{input});
  auto target = device.buffer<std::int32_t>(input.size());
  if (!source || !target) {
    return false;
  }
  std::optional<Run> retained;
  {
    auto compiled = on(device)
                        .map<std::int32_t>("run-program-lifetime", input.size(),
                                           [](auto value) { return value + 1; })
                        .compile();
    if (!compiled) {
      return false;
    }
    auto run = compiled->run(*source, *target);
    if (!run) {
      return false;
    }
    retained.emplace(std::move(run).value());
  }
  std::array<std::int32_t, 4u> output{};
  return retained->read(*target, std::span<std::int32_t>{output}) &&
         output == std::array<std::int32_t, 4u>{2, 3, 4, 5};
}

} // namespace rund_node_test_compute_reuse
