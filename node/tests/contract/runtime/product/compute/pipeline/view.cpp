#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstdint>
#include <span>

namespace runtime_compute_pipeline {

int View(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto source = device.upload<std::int32_t>(source_values);
  auto target = device.buffer<std::int32_t>(3u);
  auto reduce =
      on(device)
          .input<std::int32_t>(4u)
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  if (!source || !target || !reduce) {
    return 1;
  }
  auto input = source->view(1u, 4u, 2u);
  auto output = target->view(1u, 1u, 2u);
  if (!input || !output) {
    return 2;
  }
  auto prepared =
      pipeline(device).then(*reduce, read(*input), write(*output)).prepare();
  if (!prepared) {
    return 3;
  }
  const Completion completion = session.compute(*prepared).submit().wait();
  std::array<std::int32_t, 3u> observed{};
  if (!completion || completion.stats().internal_roundtrip_bytes != 16u ||
      !ReadExact(*prepared, *target, std::span<std::int32_t>{observed}) ||
      observed != std::array<std::int32_t, 3u>{0, 16, 0}) {
    return 4;
  }
  return 0;
}

} // namespace runtime_compute_pipeline
