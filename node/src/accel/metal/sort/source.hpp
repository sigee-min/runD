#pragma once

#include <kernel/core/model.hpp>

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string MetalSortSource(rund::kernel::u32 block_size);

}  // namespace rund::node::accel::detail
