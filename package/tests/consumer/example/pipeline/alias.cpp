#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckMultiOutputAlias(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input{3, 6, 9, 12};
  auto program = rund::compute::on(device)
                     .map<std::int32_t>("pipeline-alias", input.size(),
                                        [](auto value) { return value; })
                     .branch([](auto values) {
                       const auto doubled =
                           values.map("pipeline-alias-double",
                                      [](auto value) { return value * 2; });
                       return rund::compute::outputs(values, doubled, values);
                     })
                     .compile();
  auto source = device.upload<std::int32_t>(input);
  auto values = device.buffer<std::int32_t>(input.size());
  auto doubled = device.buffer<std::int32_t>(input.size());
  if (!program) {
    return program.exit_code();
  }
  if (!source) {
    return source.exit_code();
  }
  if (!values) {
    return values.exit_code();
  }
  if (!doubled) {
    return doubled.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*source),
                            rund::compute::write(*values, *doubled, *values))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  rund::compute::Pipeline pipeline = std::move(prepared).value();
  const auto ran = pipeline.run();
  if (!ran) {
    return ran.exit_code();
  }
  std::array<std::int32_t, input.size()> value_result{};
  std::array<std::int32_t, input.size()> doubled_result{};
  const auto value_read = pipeline.read(*values, value_result);
  const auto doubled_read = pipeline.read(*doubled, doubled_result);
  if (!value_read) {
    return value_read.exit_code();
  }
  if (!doubled_read) {
    return doubled_read.exit_code();
  }
  return value_result == input &&
                 doubled_result == std::array<std::int32_t, 4u>{6, 12, 18, 24}
             ? 0
             : 2;
}

} // namespace rund::package_example::pipeline
