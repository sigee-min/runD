#pragma once

#include <cstdint>
#include <string_view>

#include "../../hash/fnv.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] constexpr std::uint64_t
SourceHash(const std::string_view source) noexcept {
  ::rund::node::hash_detail::Fnv hash{
      ::rund::node::hash_detail::kFnvStandardOffset};
  for (const unsigned char byte : source) {
    hash.Byte(byte);
  }
  return hash.Finish();
}

}  // namespace rund::node::accel::detail
