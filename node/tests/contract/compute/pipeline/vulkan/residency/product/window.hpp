#pragma once

#include "../local.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] bool
ProductStagedLoop(std::size_t epochs,
                  ProductFault fault = ProductFault::None);

} // namespace rund_node_test_pipeline
