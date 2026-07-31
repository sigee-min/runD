#pragma once

#include <rund/host/hash.hpp>
#include <node/runtime/replay/host/archive.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace rund::node::replay_detail::payload {

class ByteHash;

[[nodiscard]] std::vector<std::byte>
EncodeRle(std::span<const std::byte> bytes);

[[nodiscard]] std::optional<std::vector<std::byte>>
DecodeRle(std::span<const std::byte> encoded,
          std::uint64_t expected_uncompressed_bytes);

[[nodiscard]] bool DecodeRleInto(std::span<const std::byte> encoded,
                                 std::span<std::byte> output,
                                 ByteHash *chunk_hash = nullptr,
                                 ByteHash *record_hash = nullptr) noexcept;

[[nodiscard]] bool RleMatches(std::span<const std::byte> encoded,
                              std::span<const std::byte> expected) noexcept;

[[nodiscard]] bool RleMatchesAndHash(std::span<const std::byte> encoded,
                                     std::span<const std::byte> expected,
                                     std::uint64_t expected_hash,
                                     ByteHash *record_hash = nullptr) noexcept;

[[nodiscard]] bool Verify(::rund::StableHash expected_hash,
                          std::uint64_t uncompressed_bytes, Codec codec,
                          std::span<const std::byte> encoded,
                          ByteHash *payload_hash = nullptr) noexcept;

[[nodiscard]] bool
AppendDecodedBytes(std::uint64_t uncompressed_bytes, Codec codec,
                   std::span<const std::byte> encoded,
                   ByteHash &payload_hash) noexcept;

} // namespace rund::node::replay_detail::payload
