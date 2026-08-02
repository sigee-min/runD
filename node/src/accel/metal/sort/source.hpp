#pragma once

#include <kernel/core/model.hpp>

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string MetalSortSource(rund::kernel::u32 block_size);
[[nodiscard]] bool MetalSortSourceUpperBytes(rund::kernel::u32 block_size,
                                             std::uint64_t &upper) noexcept;

}  // namespace rund::node::accel::detail
