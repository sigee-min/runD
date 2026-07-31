#pragma once

#include <vector>

#include "model.hpp"

namespace rund::node {

[[nodiscard]] bool ReactorReadyPrecedes(const ReactorReady &lhs,
                                        const ReactorReady &rhs) noexcept;

} // namespace rund::node
