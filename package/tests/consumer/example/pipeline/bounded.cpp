#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckBounded(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 5u> input{5, 1, 4, 2, 3};
  auto program = rund::compute::on(device)
                     .map<std::int32_t>("pipeline-bounded", input.size(),
                                        [](auto value) { return value; })
                     .filter([](auto value) { return value > 2; })
                     .compile();
  auto source = device.upload<std::int32_t>(input);
  auto values = device.buffer<std::int32_t>(input.size());
  auto count = device.buffer<std::uint32_t>(1u);
  if (!program) {
    return program.exit_code();
  }
  if (!source) {
    return source.exit_code();
  }
  if (!values) {
    return values.exit_code();
  }
  if (!count) {
    return count.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*source),
                            rund::compute::write(*values, *count))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  rund::compute::Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<std::int32_t, input.size()> filtered{};
  std::array<std::uint32_t, 1u> logical{};
  const auto values_read = pipeline.read(*values, filtered);
  const auto count_read = pipeline.read(*count, logical);
  if (!values_read) {
    return values_read.exit_code();
  }
  if (!count_read) {
    return count_read.exit_code();
  }
  return logical[0] == 3u && filtered[0] == 5 && filtered[1] == 4 &&
                 filtered[2] == 3
             ? 0
             : 2;
}

} // namespace rund::package_example::pipeline
