#pragma once

#include <kernel/program/compute/dsl.hpp>
#include <cstddef>
#include <cstdint>

namespace node_accel_contract::vulkan {

constexpr std::uint64_t kOneMiB = 1024u * 1024u;
constexpr std::size_t kStagedTileCount = 96u;

struct TileValue {
  rund::kernel::u32 value = 0u;
};

struct TileValue64 {
  rund::kernel::u64 value = 0u;
};

}  // namespace node_accel_contract::vulkan
