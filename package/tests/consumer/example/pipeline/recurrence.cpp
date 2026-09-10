#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckRecurrence(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> seed{1, 2, 3, 4};
  auto body = rund::compute::on(device)
                  .map<std::int32_t>("pipeline-repeat", seed.size(),
                                     [](auto value) { return value + 3; })
                  .compile();
  auto input = device.upload<std::int32_t>(seed);
  auto output = device.buffer<std::int32_t>(seed.size());
  auto history = device.buffer<std::int32_t>(8u * seed.size());
  if (!body || !input || !output || !history) {
    return 1;
  }
  auto prepared = rund::compute::pipeline(device)
                      .profile(rund::compute::PipelineProfile::Steps)
                      .repeat<8u>(*body, rund::compute::read(*input),
                                  rund::compute::write_final(*output))
                      .prepare();
  if (!prepared || !prepared->run()) {
    return 2;
  }
  std::array<std::int32_t, seed.size()> actual{};
  std::array<rund::compute::PipelineStepProfile, 8u> rows{};
  const auto read = prepared->read(*output, actual);
  const auto profile = prepared->profile(rows);
  if (!read || !profile ||
      actual != std::array<std::int32_t, 4u>{25, 26, 27, 28} ||
      prepared->stats().pipeline.step_count != 1u ||
      profile->written != rows.size() || profile->total != rows.size()) {
    return 3;
  }
  for (std::size_t index = 0u; index < rows.size(); ++index) {
    if (rows[index].index != 0u || rows[index].iteration != index ||
        rows[index].iteration_bound != rows.size()) {
      return 4;
    }
  }

  auto lossless = rund::compute::pipeline(device)
                      .repeat<8u>(*body, rund::compute::read(*input),
                                  rund::compute::write_each(*history))
                      .prepare();
  std::array<std::int32_t, 8u * seed.size()> each{};
  if (!lossless || !lossless->run() || !lossless->read(*history, each)) {
    return 5;
  }
  for (std::size_t iteration = 0u; iteration < 8u; ++iteration) {
    for (std::size_t element = 0u; element < seed.size(); ++element) {
      if (each[iteration * seed.size() + element] !=
          seed[element] + 3 * static_cast<std::int32_t>(iteration + 1u)) {
        return 6;
      }
    }
  }
  return 0;
}

} // namespace rund::package_example::pipeline
