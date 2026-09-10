#pragma once

#include "local.hpp"

#include <vector>

namespace rund_node_test_pipeline::view {

template <typename T>
[[nodiscard]] bool CheckGatherPreflight(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t source_count = 4096u;
  std::vector<T> values(source_count);
  for (std::size_t i = 0u; i < values.size(); ++i) {
    values[i] = static_cast<T>(i * 2654435761u);
  }
  auto source = device.upload<T>(values);
  if (!source) {
    return false;
  }
  for (const std::size_t count : {65536u, 65537u, 1048579u}) {
    std::vector<std::uint32_t> indices(count);
    std::vector<T> expected(count);
    std::vector<T> observed(count);
    for (std::size_t i = 0u; i < count; ++i) {
      indices[i] = static_cast<std::uint32_t>((i * 31u) % source_count);
      expected[i] = values[indices[i]];
    }
    auto targets = device.upload<std::uint32_t>(indices);
    auto output = device.buffer<T>(count);
    auto program =
        on(device)
            .input<T>(source_count)
            .template zip_input<std::uint32_t>(count)
            .branch([](auto input, auto index) { return input.gather(index); })
            .compile();
    if (!targets || !output || !program) {
      return false;
    }
    auto execution =
        pipeline(device)
            .then(*program, read(*source, *targets), write(*output))
            .prepare();
    if (!execution) {
      return false;
    }
    for (unsigned repeat = 0u; repeat < 3u; ++repeat) {
      if (!execution->run() ||
          !execution->read(*output, std::span<T>{observed}) ||
          observed != expected) {
        return false;
      }
    }
    // At G=257 the terminal reduction must also consume partial slot 256.
    const std::size_t first_invalid = count > 65537u ? 65536u : count - 257u;
    indices[first_invalid] = source_count;
    indices.back() = source_count;
    auto bad_targets = device.upload<std::uint32_t>(indices);
    auto bad_output = device.buffer<T>(count);
    if (!bad_targets || !bad_output) {
      return false;
    }
    auto rejected =
        pipeline(device)
            .then(*program, read(*source, *bad_targets), write(*bad_output))
            .prepare();
    if (!rejected) {
      return false;
    }
    const auto failure = rejected->run();
    if (failure || failure.error() != "compute_gather_index_out_of_range" ||
        rejected->stats().control.overflow_ordinal != first_invalid) {
      return false;
    }
  }
  return true;
}

} // namespace rund_node_test_pipeline::view
