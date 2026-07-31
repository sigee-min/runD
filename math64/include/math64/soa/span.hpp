#pragma once

#include <math64/core/model.hpp>

#include <span>

namespace rund::math64::soa {

using I64View = std::span<const i64>;
using I64MutView = std::span<i64>;
using U64View = std::span<const u64>;
using U64MutView = std::span<u64>;
using I8View = std::span<const i8>;
using U8View = std::span<const u8>;
using I16View = std::span<const i16>;

}  // namespace rund::math64::soa
