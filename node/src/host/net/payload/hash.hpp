#pragma once

#include <rund/host/hash.hpp>
#include <rund/net/vectored.hpp>

#include "../../../runtime/platform/net.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::net {

[[nodiscard]] ::rund::StableHash
PayloadHashForNative(const node::NativeCallResult &native,
                     const std::byte *data, std::uint64_t requested) noexcept;
[[nodiscard]] ::rund::StableHash
PayloadHashForPrefix(std::span<const batch::Slice> slices,
                     std::uint64_t completed_bytes) noexcept;

} // namespace rund::net
