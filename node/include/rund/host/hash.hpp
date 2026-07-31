#pragma once

#include <rund/hash.hpp>

#include <cstddef>

namespace rund::host {

// A null non-empty range is framed with its size and never dereferenced.
// Null and non-null empty ranges have the same identity.
[[nodiscard]] StableHash hash_bytes(const std::byte *data,
                                    std::size_t size) noexcept;
[[nodiscard]] StableHash hash_string(const char *data,
                                     std::size_t size) noexcept;

} // namespace rund::host
