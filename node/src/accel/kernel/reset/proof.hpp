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

[[nodiscard]] Result Prove(Spec spec, std::uint64_t target_bytes) noexcept;

} // namespace rund::node::accel::detail::reset
