#pragma once

#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] std::shared_ptr<void> RetainMetalObject(void* object);

}  // namespace rund::node::accel::detail
