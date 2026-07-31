#pragma once

#include <math32/core/model.hpp>

#include <span>

namespace rund::math32::soa {

using I32View = std::span<const i32>;
using I32MutView = std::span<i32>;
using U32View = std::span<const u32>;
using U32MutView = std::span<u32>;
using I8View = std::span<const i8>;
using U8View = std::span<const u8>;
using I16View = std::span<const i16>;

}  // namespace rund::math32::soa
