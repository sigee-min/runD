#pragma once

#include <rund/host/hash.hpp>
#include <rund/net/address.hpp>

namespace rund::net {

[[nodiscard]] ::rund::StableHash HashAddress(Address address) noexcept;

} // namespace rund::net
