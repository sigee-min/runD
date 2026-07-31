#include "src/compute/cpu/view.hpp"
#include "src/compute/size.hpp"

#include <rund/compute.hpp>

#include <limits>

namespace node_compute_cpu {

int CheckViewFootprint() noexcept {
  using rund::compute::detail::cpu_footprint;
  using rund::compute::detail::size::add;
  using rund::compute::detail::size::multiply;
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();

  std::size_t checked = 37u;
  if (!add(0u, 0u, checked) || checked != 0u || !add(0u, 1u, checked) ||
      checked != 1u || !add(1u, maximum - 1u, checked) || checked != maximum ||
      !add(maximum, 0u, checked) || checked != maximum ||
      add(maximum, 1u, checked) || checked != maximum ||
      !multiply(0u, maximum, checked) || checked != 0u ||
      !multiply(1u, maximum, checked) || checked != maximum ||
      !multiply(maximum, 1u, checked) || checked != maximum ||
      multiply(maximum, 2u, checked) || checked != maximum ||
      !multiply(maximum, 0u, checked) || checked != 0u ||
      !multiply(2u, 3u, 4u, checked) || checked != 24u ||
      multiply(maximum, 2u, 0u, checked) || checked != 24u ||
      multiply(maximum, 1u, 2u, checked) || checked != 24u) {
    return 1;
  }

  if (cpu_footprint(64u, 0u, 1u, 1u, 0u) ||
      cpu_footprint(64u, 0u, 1u, 0u, 1u) ||
      cpu_footprint(maximum, maximum / 2u + 1u, 0u, 1u, 2u)) {
    return 2;
  }

  const auto empty = cpu_footprint(maximum, maximum, 0u, maximum, 1u);
  if (!empty || empty->base != maximum || empty->stride != 1u ||
      empty->bytes != 0u || empty->count != 0u || empty->width != 1u ||
      !empty->dense()) {
    return 3;
  }

  const auto single = cpu_footprint(64u, 15u, 1u, maximum, 4u);
  if (!single || single->base != 60u || single->stride != 4u ||
      single->bytes != 4u || single->count != 1u || single->width != 4u ||
      !single->dense() || cpu_footprint(63u, 15u, 1u, maximum, 4u)) {
    return 4;
  }

  const auto widest = cpu_footprint(maximum, 0u, 1u, 1u, maximum);
  if (!widest || widest->base != 0u || widest->stride != maximum ||
      widest->bytes != maximum || widest->width != maximum ||
      !widest->dense() || cpu_footprint(maximum - 1u, 0u, 1u, 1u, maximum)) {
    return 5;
  }

  const auto longest = cpu_footprint(maximum, 0u, 2u, maximum - 1u, 1u);
  if (!longest || longest->base != 0u || longest->stride != maximum - 1u ||
      longest->bytes != 2u || longest->count != 2u || longest->width != 1u ||
      longest->dense() || cpu_footprint(maximum, 0u, 2u, maximum, 1u)) {
    return 6;
  }

  const auto strided = cpu_footprint(68u, 1u, 4u, 5u, 4u);
  if (!strided || strided->base != 4u || strided->stride != 20u ||
      strided->bytes != 16u || strided->count != 4u || strided->width != 4u ||
      strided->dense() || cpu_footprint(67u, 1u, 4u, 5u, 4u)) {
    return 7;
  }

  const auto dense = cpu_footprint(20u, 2u, 3u, 1u, 4u);
  if (!dense || dense->base != 8u || dense->stride != 4u ||
      dense->bytes != 12u || dense->count != 3u || dense->width != 4u ||
      !dense->dense()) {
    return 8;
  }

  const auto full = cpu_footprint(maximum, 0u, maximum, 1u, 1u);
  if (!full || full->base != 0u || full->stride != 1u ||
      full->bytes != maximum || full->count != maximum || full->width != 1u ||
      !full->dense()) {
    return 9;
  }

  for (std::size_t storage = 0u; storage <= 32u; ++storage) {
    for (std::size_t offset = 0u; offset <= 8u; ++offset) {
      for (std::size_t count = 0u; count <= 8u; ++count) {
        for (std::size_t stride = 0u; stride <= 8u; ++stride) {
          for (std::size_t width = 0u; width <= 8u; ++width) {
            const std::size_t base = offset * width;
            const std::size_t end =
                count == 0u ? base
                            : (offset + (count - 1u) * stride) * width + width;
            const bool expected = width != 0u && stride != 0u &&
                                  base <= storage && end <= storage;
            const auto observed =
                cpu_footprint(storage, offset, count, stride, width);
            if (static_cast<bool>(observed) != expected) {
              return 10;
            }
            if (expected &&
                (observed->base != base ||
                 observed->stride != (count <= 1u ? width : stride * width) ||
                 observed->bytes != count * width || observed->count != count ||
                 observed->width != width ||
                 observed->dense() != (count <= 1u || stride == 1u))) {
              return 11;
            }
          }
        }
      }
    }
  }
  return 0;
}

int CheckCanonicalView() {
  using namespace rund::compute;
  auto device = open(Target::cpu(1u));
  auto buffer =
      device ? device->buffer<std::uint64_t>(2u)
             : Result<Buffer<std::uint64_t>>::fail(Reason::DeviceInvalid);
  if (!buffer) {
    return 1;
  }
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  auto empty = buffer->view(2u, 0u, maximum);
  auto single = buffer->view(1u, 1u, maximum);
  const Buffer<std::uint64_t> &read_only = *buffer;
  auto constant = read_only.view(0u, 1u, maximum);
  if (!empty || !single || !constant || empty->stride() != 1u ||
      single->stride() != 1u || constant->stride() != 1u ||
      empty->stride_bytes() != sizeof(std::uint64_t) ||
      single->stride_bytes() != sizeof(std::uint64_t) ||
      constant->stride_bytes() != sizeof(std::uint64_t) ||
      empty->span_bytes() != 0u ||
      single->span_bytes() != sizeof(std::uint64_t) ||
      constant->span_bytes() != sizeof(std::uint64_t)) {
    return 2;
  }
  return 0;
}

} // namespace node_compute_cpu
