#pragma once

#include <cstdint>

namespace rund::compute {

enum class Code : std::uint8_t {
  Ok,
  Invalid,
  Unsupported,
  Unavailable,
  Capacity,
  Compile,
  Binding,
  Transfer,
  Execution,
};

enum class Reason : std::uint16_t {
#define RUND_COMPUTE_REASON(name, value, text) name = value,
#include <rund/compute/reason.def>
#undef RUND_COMPUTE_REASON
};

namespace detail {

[[nodiscard]] constexpr Code category(const Reason reason) noexcept {
  return static_cast<Code>(static_cast<std::uint16_t>(reason) >> 12u);
}

[[nodiscard]] constexpr std::uint16_t ordinal(const Reason reason) noexcept {
  return static_cast<std::uint16_t>(static_cast<std::uint16_t>(reason) &
                                    0x0fffu);
}

[[nodiscard]] constexpr bool valid(const Reason reason) noexcept {
  const auto value = ordinal(reason);
  if (value == 0u) {
    return false;
  }
  switch (category(reason)) {
  case Code::Invalid:
    return value <= ordinal(Reason::ScatterReduceIndexOutOfRange);
  case Code::Unsupported:
    return value <= ordinal(Reason::BatchCpuUnsupported);
  case Code::Unavailable:
    return value <= ordinal(Reason::ProfileUnavailable);
  case Code::Capacity:
    return value <= ordinal(Reason::PipelineMemoryBudget);
  case Code::Compile:
    return value <= ordinal(Reason::ProgramCompileException);
  case Code::Binding:
    return value <= ordinal(Reason::ScatterReduceCountOutOfRange);
  case Code::Transfer:
    return value <= ordinal(Reason::TransferInvalid);
  case Code::Execution:
    return value <= ordinal(Reason::DeviceLost);
  case Code::Ok:
    return false;
  }
  return false;
}

} // namespace detail

} // namespace rund::compute
