#pragma once

#include "model.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>

namespace rund::node::accel::detail::reset {

[[nodiscard]] Spec Project(const rund::kernel::ResidentBufferRef &source,
                           const Replacement *replacement) noexcept;

[[nodiscard]] Range Prove(Spec spec, std::uint64_t target_bytes) noexcept;

} // namespace rund::node::accel::detail::reset
