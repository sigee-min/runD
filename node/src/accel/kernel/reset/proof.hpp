#pragma once

#include "model.hpp"

#include <accel/check.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>

namespace rund::node::accel::detail::reset {

struct Result final {
  rund::AccelCheck check{};
  Range range{};
};

[[nodiscard]] Spec Project(const rund::kernel::ResidentBufferRef &source,
                           const Replacement *replacement) noexcept;

// word_limit is the largest admitted exclusive 32-bit word extent. Metal
// supplies uint64_t maximum; Vulkan strided reset supplies its shader's
// uint32_t address limit.
[[nodiscard]] Result Prove(Spec spec, std::uint64_t target_bytes,
                           std::uint64_t word_limit) noexcept;

} // namespace rund::node::accel::detail::reset
