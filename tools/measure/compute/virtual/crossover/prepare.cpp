#include "internal.hpp"

#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace rund::measure::compute::virtual_crossover::detail {

std::size_t active_count(const std::size_t logical_count,
                         const ActiveRatio ratio) noexcept {
  return logical_count * ratio.numerator / ratio.denominator;
}

std::size_t frame_elements(const std::size_t radius) noexcept {
  return CorePageElements + radius * 2u;
}

std::uint64_t input_frame_bytes(const std::size_t radius) noexcept {
  return frame_elements(radius) * sizeof(std::int32_t);
}

::rund::compute::ResidencyConfig
residency_config(const Backend backend, const std::size_t radius) noexcept {
  const std::uint64_t frame = input_frame_bytes(radius);
  const std::uint64_t pair = frame * 2u;
  return {
      .device_resident_bytes = pair * RequestedDeviceFrames * 2u,
      .host_resident_bytes =
          backend == Backend::Cpu
              ? pair * RequestedDeviceFrames * 2u
              : (frame * RequestedHostFrames + frame * RequestedDeviceFrames) *
                    2u,
  };
}

constexpr std::uint64_t BaseFrameBytes =
    CorePageElements * sizeof(std::int32_t);
static_assert(BaseFrameBytes * 2u * RequestedDeviceFrames * 2u == 196'608u);
static_assert((BaseFrameBytes * RequestedHostFrames +
               BaseFrameBytes * RequestedDeviceFrames) *
                  2u ==
              491'520u);

std::int32_t seed_value(const std::size_t index) noexcept {
  return static_cast<std::int32_t>((index * 17u + 11u) % 63u) - 31;
}

void seed(const std::span<std::int32_t> values) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = seed_value(index);
  }
}

PreparedPoint::PreparedPoint(
    const Backend selected, const std::size_t count,
    const std::size_t selected_radius,
    std::shared_ptr<virtual_residency::MemoryBacking> input_owner,
    std::shared_ptr<virtual_residency::MemoryBacking> output_owner,
    Pipeline value) noexcept
    : backend(selected), logical_count(count), radius(selected_radius),
      input_backing(std::move(input_owner)),
      output_backing(std::move(output_owner)), pipeline(std::move(value)),
      plan(pipeline.plan()), observed(count) {}

std::unique_ptr<PreparedPoint> prepare(const Backend backend,
                                       const std::size_t logical_count,
                                       const std::size_t radius) noexcept {
  try {
    auto device = ::rund::compute::open(TargetFor(backend));
    if (!device) {
      return nullptr;
    }
    auto flow = ::rund::compute::on(*device).input<std::int32_t>(
        frame_elements(radius));
    auto program = std::move(flow)
                       .branch([radius](auto values) {
                         return values.window(::rund::compute::WindowSpec{
                             .op = ::rund::compute::Window::Sum,
                             .radius = radius,
                             .edge = ::rund::compute::WindowEdge::Clamp});
                       })
                       .compile();
    auto input_backing = std::make_shared<virtual_residency::MemoryBacking>(
        logical_count * sizeof(std::int32_t));
    auto output_backing = std::make_shared<virtual_residency::MemoryBacking>(
        logical_count * sizeof(std::int32_t));
    std::vector<std::int32_t> seeded(logical_count);
    seed(seeded);
    const auto seeded_status =
        input_backing->write(0u, std::as_bytes(std::span{seeded}));
    auto input = ::rund::compute::virtual_buffer<std::int32_t>(logical_count,
                                                               input_backing);
    auto output = ::rund::compute::virtual_buffer<std::int32_t>(logical_count,
                                                                output_backing);
    auto prepared =
        program && input && output
            ? ::rund::compute::virtual_pipeline(
                  *program, *input, *output, residency_config(backend, radius))
            : decltype(::rund::compute::virtual_pipeline(
                  *program, *input, *output,
                  residency_config(
                      backend,
                      radius)))::fail(::rund::compute::Reason::PipelineInvalid);
    if (!seeded_status || !program || !input || !output || !prepared) {
      return nullptr;
    }
    auto point = std::make_unique<PreparedPoint>(
        backend, logical_count, radius, std::move(input_backing),
        std::move(output_backing), std::move(prepared).value());
    const auto preparation = point->pipeline.profile();
    if (!preparation || !valid_preparation(*preparation, backend)) {
      return nullptr;
    }
    return point;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

} // namespace rund::measure::compute::virtual_crossover::detail
