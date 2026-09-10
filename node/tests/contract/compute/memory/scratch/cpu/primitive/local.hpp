#pragma once

#include "../../../local.hpp"

namespace rund_node_memory_contract::cpu_primitive {

[[nodiscard]] int CheckRange();
[[nodiscard]] int CheckEmpty();
[[nodiscard]] int CheckScatter();
[[nodiscard]] int CheckTypedI32();
[[nodiscard]] int CheckTypedI64();

} // namespace rund_node_memory_contract::cpu_primitive
