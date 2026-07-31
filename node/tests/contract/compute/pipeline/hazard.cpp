#include "local.hpp"

#include <utility>
#include <vector>

namespace rund_node_test_pipeline {

// This leaf owns fresh buffers; no result depends on another contract leaf.
[[nodiscard]] int CheckHazards(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> first_values{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> second_values{5, 6, 7, 8};
  auto add = on(device)
                 .map<std::int32_t>("pipeline-hazard-add", first_values.size(),
                                    [](auto value) { return value + 10; })
                 .compile();
  auto multiply =
      on(device)
          .map<std::int32_t>("pipeline-hazard-multiply", first_values.size(),
                             [](auto value) { return value * 2; })
          .compile();
  auto a = Upload(device, first_values);
  auto c = Upload(device, second_values);
  auto b = device.buffer<std::int32_t>(first_values.size());
  auto d = device.buffer<std::int32_t>(first_values.size());
  if (!add || !multiply || !a || !b || !c || !d) {
    return 1;
  }

  auto raw = pipeline(device)
                 .then(*add, read(*a), write(*b))
                 .then(*multiply, read(*b), write(*d))
                 .prepare();
  if (!raw || !raw->run() || raw->stats().pipeline.barrier_count != 1u) {
    return 2;
  }
  std::array<std::int32_t, 4u> observed{};
  if (!ReadExact(*raw, *d, observed) ||
      observed != std::array<std::int32_t, 4u>{22, 24, 26, 28}) {
    return 3;
  }

  auto war = pipeline(device)
                 .then(*add, read(*a), write(*b))
                 .then(*multiply, read(*c), write(*a))
                 .prepare();
  if (!war || !war->run() || war->stats().pipeline.barrier_count != 1u ||
      !ReadExact(*war, *b, observed) ||
      observed != std::array<std::int32_t, 4u>{11, 12, 13, 14} ||
      !ReadExact(*war, *a, observed) ||
      observed != std::array<std::int32_t, 4u>{10, 12, 14, 16}) {
    return 4;
  }

  auto restored_a = Upload(device, first_values);
  if (!restored_a) {
    return 5;
  }
  auto waw = pipeline(device)
                 .then(*add, read(*restored_a), write(*b))
                 .then(*multiply, read(*c), write(*b))
                 .prepare();
  if (!waw || !waw->run() || waw->stats().pipeline.barrier_count != 1u ||
      !ReadExact(*waw, *b, observed) ||
      observed != std::array<std::int32_t, 4u>{10, 12, 14, 16}) {
    return 6;
  }

  auto rr = pipeline(device)
                .then(*add, read(*restored_a), write(*b))
                .then(*multiply, read(*restored_a), write(*d))
                .prepare();
  if (!rr || !rr->run() || rr->stats().pipeline.barrier_count != 0u ||
      !ReadExact(*rr, *b, observed) ||
      observed != std::array<std::int32_t, 4u>{11, 12, 13, 14} ||
      !ReadExact(*rr, *d, observed) ||
      observed != std::array<std::int32_t, 4u>{2, 4, 6, 8}) {
    return 7;
  }

  // Exercise the last row/column of the bounded 64 x 64 dependency-witness
  // index.  Distinct intermediates keep the exact hazard set to the 63
  // adjacent RAW edges instead of manufacturing a repeated-owner quadratic
  // fixture.
  auto chain_source = Upload(device, first_values);
  if (!chain_source) {
    return 8;
  }
  std::vector<Buffer<std::int32_t>> chain;
  chain.reserve(65u);
  chain.push_back(std::move(*chain_source));
  for (std::size_t index = 0u; index < 64u; ++index) {
    auto buffer = device.buffer<std::int32_t>(first_values.size());
    if (!buffer) {
      return 9;
    }
    chain.push_back(std::move(*buffer));
  }
  auto boundary_builder = pipeline(device);
  for (std::size_t index = 0u; index < 64u; ++index) {
    boundary_builder.then(*add, read(chain[index]), write(chain[index + 1u]));
  }
  auto boundary = std::move(boundary_builder).prepare();
  if (!boundary || !boundary->run() ||
      boundary->stats().pipeline.step_count != 64u ||
      boundary->stats().pipeline.resource_count != 65u ||
      boundary->stats().pipeline.barrier_count != 63u ||
      !ReadExact(*boundary, chain.back(), observed) ||
      observed != std::array<std::int32_t, 4u>{641, 642, 643, 644}) {
    return 10;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
