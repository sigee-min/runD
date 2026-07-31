#pragma once

#include "catalog/entry.hpp"

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice
PickFromCatalog(std::span<const BackendEntry> catalog,
                const rund::AccelPolicy &policy);

} // namespace rund::node::accel::detail
