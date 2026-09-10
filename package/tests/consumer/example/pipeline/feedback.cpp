#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <span>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckHostFeedback(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 2u> seed{1, 4};
  auto body = rund::compute::on(device)
                  .map<std::int32_t>("pipeline-host-feedback", seed.size(),
                                     [](auto value) { return value + 1; })
                  .compile();
  auto input = device.upload<std::int32_t>(seed);
  auto output = device.buffer<std::int32_t>(seed.size());
  if (!body || !input || !output) {
    return 1;
  }
  auto prepared = rund::compute::pipeline(device)
                      .then(*body, rund::compute::read(*input),
                            rund::compute::write(*output))
                      .prepare();
  if (!prepared) {
    return 2;
  }
  std::array<std::int32_t, seed.size()> observed{};
  const auto status = rund::compute::host_feedback(
      *prepared, 3u,
      [&](rund::compute::HostIteration &step) noexcept
          -> rund::compute::Status {
        if (auto read = step.read(*output, observed); !read) {
          return read;
        }
        return step.has_next()
                   ? step.write(*input, std::span<const std::int32_t>{observed})
                   : rund::compute::Status::success();
      });
  return status && observed == std::array<std::int32_t, 2u>{4, 7} ? 0 : 3;
}

} // namespace rund::package_example::pipeline
