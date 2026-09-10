#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckRangeIntrospection(rund::compute::Device &device,
                                          const Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t count = 515u;
  constexpr std::size_t radius = 257u;
  std::array<std::uint32_t, count> values{};
  values.fill(1u);

  auto program = on(device)
                     .input<std::uint32_t>(count)
                     .branch([](auto input) {
                       return input.window(
                           WindowSpec{.op = Window::Sum, .radius = radius});
                     })
                     .compile();
  auto input =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{values});
  auto output = device.buffer<std::uint32_t>(count);
  if (!program || !input || !output) {
    return 1;
  }

  const RangeSnapshot empty = program->ranges({});
  std::array<RangeInfo, 2u> rows{};
  const RangeSnapshot full = program->ranges(rows);
  const RangeInfo frozen = rows[0u];
  const bool physical =
      backend == Backend::Cpu
          ? frozen.kind == RangeKind::Prefix && frozen.stages == 2u &&
                frozen.scratch_bytes != 0u
          : frozen.kind == RangeKind::Tiled && frozen.stages == 1u &&
                frozen.scratch_bytes == 0u;
  if (empty.written != 0u || empty.total != 1u || !empty.truncated() ||
      full.written != 1u || full.total != 1u || full.truncated() || !physical ||
      frozen.shared_capacity != 0u || !frozen.source || !frozen.execution ||
      (frozen.width != 64u && frozen.width != 128u && frozen.width != 256u)) {
    std::fprintf(stderr,
                 "pipeline range backend=%u empty=%zu/%zu full=%zu/%zu "
                 "kind=%u width=%u stages=%u shared=%u scratch=%llu\n",
                 static_cast<unsigned>(backend), empty.written, empty.total,
                 full.written, full.total, static_cast<unsigned>(frozen.kind),
                 frozen.width, frozen.stages, frozen.shared_capacity,
                 static_cast<unsigned long long>(frozen.scratch_bytes));
    return 2;
  }

  auto prepared =
      pipeline(device).then(*program, read(*input), write(*output)).prepare();
  std::array<std::uint32_t, count> observed{};
  if (!prepared || !prepared->run() ||
      !prepared->read(*output, std::span<std::uint32_t>{observed}) ||
      !std::ranges::all_of(
          observed, [](const std::uint32_t value) { return value == count; })) {
    return 3;
  }

  rows = {};
  const RangeSnapshot warm = program->ranges(rows);
  const RangeInfo retained = rows[0u];
  return warm.written == 1u && warm.total == 1u && !warm.truncated() &&
                 retained.node == frozen.node && retained.kind == frozen.kind &&
                 retained.width == frozen.width &&
                 retained.stages == frozen.stages &&
                 retained.shared_capacity == frozen.shared_capacity &&
                 retained.scratch_bytes == frozen.scratch_bytes &&
                 retained.source == frozen.source &&
                 retained.execution == frozen.execution
             ? 0
             : 4;
}

} // namespace rund_node_test_pipeline
