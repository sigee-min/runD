#include "../internal.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::measure::compute::route_matrix {
namespace {

using Program = ::rund::compute::Program<std::int32_t(std::int32_t)>;

[[nodiscard]] ::rund::compute::Result<Program>
build_program(::rund::compute::Device &device, const CaseSpec &spec,
              const Clock::time_point begin, ColdTiming &timing) {
  if (spec.family == "spatial_window") {
    auto flow = ::rund::compute::on(device).input<std::int32_t>(
        PageElements + 2u * static_cast<std::size_t>(spec.window));
    auto selected = std::move(flow).branch([radius = spec.window](auto value) {
      return value.window(::rund::compute::WindowSpec{
          .op = ::rund::compute::Window::Sum,
          .radius = static_cast<std::size_t>(radius),
          .edge = ::rund::compute::WindowEdge::Clamp});
    });
    const auto authored = Clock::now();
    timing.author_us = micros(authored - begin);
    auto program = std::move(selected).compile();
    const auto compiled = Clock::now();
    timing.compile_us = micros(compiled - authored);
    return program;
  }
  auto flow = ::rund::compute::on(device).map<std::int32_t>(
      "route-matrix-pointwise", PageElements,
      [](auto value) { return (value + 5) * 3; });
  const auto authored = Clock::now();
  timing.author_us = micros(authored - begin);
  auto program = std::move(flow).compile();
  const auto compiled = Clock::now();
  timing.compile_us = micros(compiled - authored);
  return program;
}

[[nodiscard]] bool write_seed(const std::shared_ptr<Backing> &backing,
                              const std::uint64_t count) noexcept {
  if (backing == nullptr || count == 0u ||
      count > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  try {
    std::vector<std::int32_t> values(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0u; index < count; ++index) {
      values[static_cast<std::size_t>(index)] = oracle::seed_value(index);
    }
    return static_cast<bool>(
        backing->write(0u, std::as_bytes(std::span{values})));
  } catch (const std::bad_alloc &) {
    return false;
  }
}

} // namespace

double micros(const Clock::duration duration) noexcept {
  return std::chrono::duration<double, std::micro>(duration).count();
}

std::uint64_t elements(const CaseSpec &spec) noexcept {
  if (spec.q == 0u ||
      spec.q > std::numeric_limits<std::uint64_t>::max() / PageElements) {
    return 0u;
  }
  return spec.q * PageElements - 1u;
}

std::uint64_t pages(const std::uint64_t count) noexcept {
  return count / PageElements +
         static_cast<std::uint64_t>(count % PageElements != 0u);
}

std::uint64_t frame_elements(const CaseSpec &spec) noexcept {
  if (spec.family != "spatial_window" ||
      spec.window >
          (std::numeric_limits<std::uint64_t>::max() - PageElements) / 2u) {
    return spec.family == "spatial_window" ? 0u : PageElements;
  }
  return PageElements + spec.window * 2u;
}

bool fill_tail(const std::shared_ptr<Backing> &backing,
               const std::uint64_t count) noexcept {
  if (backing == nullptr || count == 0u ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
    return false;
  }
  try {
    std::vector<std::byte> values(static_cast<std::size_t>(count) *
                                      sizeof(std::int32_t),
                                  virtual_residency::TailPoison);
    return static_cast<bool>(backing->write(0u, values));
  } catch (const std::bad_alloc &) {
    return false;
  }
}

::rund::compute::ResidencyConfig
residency_config(const CaseSpec &spec) noexcept {
  const std::uint64_t frame = frame_elements(spec);
  if (frame == 0u || frame > std::numeric_limits<std::uint64_t>::max() /
                                 sizeof(std::int32_t)) {
    return {};
  }
  const std::uint64_t frame_bytes = frame * sizeof(std::int32_t);
  if (frame_bytes > std::numeric_limits<std::uint64_t>::max() / 2u) {
    return {};
  }
  const std::uint64_t pair_bytes = frame_bytes * 2u;
  if (spec.q > std::numeric_limits<std::uint64_t>::max() - FrameCapacity) {
    return {};
  }
  const std::uint64_t host_frames = spec.q + FrameCapacity;
  if (host_frames == 0u ||
      pair_bytes > std::numeric_limits<std::uint64_t>::max() / host_frames ||
      pair_bytes > std::numeric_limits<std::uint64_t>::max() / FrameCapacity) {
    return {};
  }
  const std::uint64_t device_bytes = pair_bytes * FrameCapacity;
  if (device_bytes > std::numeric_limits<std::uint64_t>::max() / 2u) {
    return {};
  }
  return {.device_resident_bytes = device_bytes * 2u,
          .host_resident_bytes = pair_bytes * host_frames};
}

Job prepare_job(::rund::compute::Device &device, const CaseSpec &spec,
                const bool resident,
                const ::rund::compute::ResidencyConfig config,
                const bool require_config) {
  Job job{};
  job.resident = resident;
  const auto begin = Clock::now();
  const std::uint64_t count = elements(spec);
  if (count == 0u || count > std::numeric_limits<std::size_t>::max()) {
    job.reason = ::rund::compute::Reason::ShapeMismatch;
    return job;
  }

  auto program = build_program(device, spec, begin, job.cold);
  const auto compiled = Clock::now();
  if (!program) {
    job.reason = program.reason();
    return job;
  }

  try {
    if (resident) {
      auto input = ::rund::compute::resident_virtual_backing<std::int32_t>(
          device, count);
      auto output = ::rund::compute::resident_virtual_backing<std::int32_t>(
          device, count);
      if (!input || !output) {
        job.reason = !input ? input.reason() : output.reason();
        return job;
      }
      job.input = std::move(input).value();
      job.output = std::move(output).value();
    } else {
      const std::size_t bytes =
          static_cast<std::size_t>(count) * sizeof(std::int32_t);
      job.input = std::make_shared<virtual_residency::MemoryBacking>(bytes);
      job.output = std::make_shared<virtual_residency::MemoryBacking>(bytes);
    }
    if (!job.input || !job.output || !write_seed(job.input, count) ||
        !fill_tail(job.output, count)) {
      job.reason = ::rund::compute::Reason::BufferCapacity;
      return job;
    }
    const auto seeded = Clock::now();
    const auto input =
        ::rund::compute::virtual_buffer<std::int32_t>(count, job.input);
    auto output =
        ::rund::compute::virtual_buffer<std::int32_t>(count, job.output);
    if (!input || !output) {
      job.reason = !input ? input.reason() : output.reason();
      return job;
    }
    if (require_config && (config.device_resident_bytes == 0u ||
                           config.host_resident_bytes == 0u)) {
      job.reason = ::rund::compute::Reason::BufferCapacity;
      return job;
    }
    auto prepared =
        ::rund::compute::virtual_pipeline(*program, *input, *output, config);
    const auto prepared_at = Clock::now();
    job.cold.prepare_us = micros(prepared_at - compiled);
    job.cold.seed_us = micros(seeded - compiled);
    if (!prepared) {
      job.reason = prepared.reason();
      return job;
    }
    job.pipeline = std::make_unique<Pipeline>(std::move(prepared).value());
    job.plan = job.pipeline->plan();
    job.prepared = true;
    job.reason = ::rund::compute::Reason::Ok;
    return job;
  } catch (const std::bad_alloc &) {
    job.reason = ::rund::compute::Reason::BufferCapacity;
    return job;
  }
}

} // namespace rund::measure::compute::route_matrix
