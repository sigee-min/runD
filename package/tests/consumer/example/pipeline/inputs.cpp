#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckMultiInput(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> first{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> second{10, 20, 30, 40};
  constexpr std::array<std::int32_t, 4u> third{100, 200, 300, 400};

  auto program = rund::compute::on(device)
                     .input<std::int32_t>(first.size())
                     .zip_input<std::int32_t>(second.size())
                     .zip_input<std::int32_t>(third.size())
                     .map("pipeline-multi-input-sum",
                          [](auto a, auto b, auto c) { return a + b + c; })
                     .compile();
  auto a = device.upload<std::int32_t>(first);
  auto b = device.upload<std::int32_t>(second);
  auto c = device.upload<std::int32_t>(third);
  auto sum = device.buffer<std::int32_t>(first.size());
  if (!program) {
    return program.exit_code();
  }
  if (!a) {
    return a.exit_code();
  }
  if (!b) {
    return b.exit_code();
  }
  if (!c) {
    return c.exit_code();
  }
  if (!sum) {
    return sum.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*a, *b, *c),
                            rund::compute::write(*sum))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  rund::compute::Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<std::int32_t, first.size()> result{};
  const auto read = pipeline.read(*sum, result);
  if (!read) {
    return read.exit_code();
  }
  return result == std::array<std::int32_t, 4u>{111, 222, 333, 444} ? 0 : 2;
}

} // namespace rund::package_example::pipeline
