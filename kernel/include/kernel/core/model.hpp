#pragma once

#include <math32/core/model.hpp>

namespace rund::kernel {

using math32::i8;
using math32::i16;
using math32::i32;
using math32::i64;
using math32::u8;
using math32::u16;
using math32::u32;
using math32::u64;
using i128 = __int128_t;
using u128 = __uint128_t;

struct Partition {
  u32 worker_index = 0u;
  u32 begin = 0u;
  u32 end = 0u;
  const u32* packet_indices = nullptr;
  u32 packet_index_count = 0u;

  [[nodiscard]] u32 size() const { return end > begin ? (end - begin) : 0u; }
  [[nodiscard]] u32 packet_count() const {
    return packet_indices != nullptr ? packet_index_count : size();
  }
  [[nodiscard]] u32 packet_at(const u32 offset) const {
    return packet_indices != nullptr ? packet_indices[offset] : begin + offset;
  }
};

} // namespace rund::kernel
