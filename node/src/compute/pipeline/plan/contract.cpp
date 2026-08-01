#include "contract.hpp"

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool fixed_type(const Type type) noexcept {
  return type == Type::FixedLane32 || type == Type::FixedLane64;
}

} // namespace

PipelineHash::PipelineHash() noexcept {
  text("rund.compute.pipeline");
  number(3u);
}

void PipelineHash::byte(const std::uint8_t value) noexcept {
  lo_ ^= value;
  lo_ *= 1099511628211ull;
  hi_ ^= static_cast<std::uint8_t>(value + 0x9du);
  hi_ *= 14029467366897019727ull;
}

void PipelineHash::number(const std::uint64_t value) noexcept {
  for (unsigned shift = 0u; shift != 64u; shift += 8u) {
    byte(static_cast<std::uint8_t>(value >> shift));
  }
}

void PipelineHash::text(const char *value) noexcept {
  while (*value != '\0') {
    byte(static_cast<std::uint8_t>(*value++));
  }
  byte(0u);
}

void PipelineHash::format(const FixedFormat value) noexcept {
  byte(value.integer_bits);
  byte(value.fraction_bits);
  byte(static_cast<std::uint8_t>(value.rounding));
  byte(static_cast<std::uint8_t>(value.overflow));
  byte(static_cast<std::uint8_t>(value.approximation));
}

graph::Fingerprint PipelineHash::finish() const noexcept {
  return graph::Fingerprint{.hi = hi_, .lo = lo_};
}

bool valid_format(const Type type, const FixedFormat format) noexcept {
  if (!fixed_type(type)) {
    return format == FixedFormat{};
  }
  const unsigned width = static_cast<unsigned>(format.integer_bits) +
                         static_cast<unsigned>(format.fraction_bits);
  const unsigned required = type == Type::FixedLane32 ? 32u : 64u;
  return format.integer_bits != 0u && format.fraction_bits != 0u &&
         width == required &&
         static_cast<unsigned>(format.rounding) <=
             static_cast<unsigned>(Rounding::NearestEven) &&
         static_cast<unsigned>(format.overflow) <=
             static_cast<unsigned>(Overflow::Wrap) &&
         static_cast<unsigned>(format.approximation) <=
             static_cast<unsigned>(Approximation::Deterministic);
}

bool typed_format_matches(const Type type, const FixedFormat typed,
                          const FixedFormat slot) noexcept {
  if (!valid_format(type, slot)) {
    return false;
  }
  if (!fixed_type(type)) {
    return typed == FixedFormat{};
  }
  return typed.integer_bits == slot.integer_bits &&
         typed.fraction_bits == slot.fraction_bits;
}

} // namespace rund::compute::detail
