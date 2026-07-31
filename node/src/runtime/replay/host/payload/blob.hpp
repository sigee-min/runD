#pragma once

#include "codec.hpp"

#include <node/runtime/replay/host/bytes.hpp>
#include <rund/host/hash.hpp>

#include <cstdint>

namespace rund::node::replay_detail::payload {

struct Blob final {
  ::rund::StableHash payload_hash{};
  std::uint64_t uncompressed_bytes = 0u;
  std::uint64_t encoded_bytes = 0u;
  Codec codec = Codec::Raw;
  ::rund::node::replay_detail::payload::Bytes encoded{};
};

} // namespace rund::node::replay_detail::payload
