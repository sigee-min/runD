#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr std::uint32_t DeviceVsmPageMapMagic = 0x504d4150u;
inline constexpr std::uint32_t DeviceVsmPageMapVersion = 1u;
inline constexpr std::size_t DeviceVsmPageMapRowCapacity = 8u;
inline constexpr std::size_t DeviceVsmPageMapEntryCapacity = 32u;
inline constexpr std::size_t DeviceVsmPageMapHeaderWords = 8u;
inline constexpr std::size_t DeviceVsmPageMapRowWords =
    6u + 3u * DeviceVsmPageMapEntryCapacity;
inline constexpr std::size_t DeviceVsmPageMapWordCount =
    DeviceVsmPageMapHeaderWords +
    DeviceVsmPageMapRowCapacity * DeviceVsmPageMapRowWords;
inline constexpr std::size_t DeviceVsmPageMapBytes =
    DeviceVsmPageMapWordCount * sizeof(std::uint32_t);

struct DeviceVsmPageMapEntry final {
  std::uint32_t target{};
  std::uint32_t source{};
  std::uint32_t origin{};
};

struct DeviceVsmPageMapRow final {
  std::uint32_t resource{};
  std::uint32_t external_slot{};
  std::uint32_t frame_capacity{};
  std::uint64_t page_bytes{};
  std::uint32_t count{};
  std::array<DeviceVsmPageMapEntry, DeviceVsmPageMapEntryCapacity> entries{};
};

// This is the proof-owned representation and the exact native buffer image.
// No backend or caller stores a pointer into it.
struct DeviceVsmPageMap final {
  std::array<std::uint32_t, DeviceVsmPageMapWordCount> words{};
};

static_assert(DeviceVsmPageMapRowWords == 102u);
static_assert(DeviceVsmPageMapWordCount == 824u);
static_assert(DeviceVsmPageMapBytes == 3296u);
static_assert(sizeof(DeviceVsmPageMap) == DeviceVsmPageMapBytes);

[[nodiscard]] inline constexpr std::size_t
device_vsm_page_map_row_base(const std::size_t row) noexcept {
  return DeviceVsmPageMapHeaderWords + row * DeviceVsmPageMapRowWords;
}

[[nodiscard]] inline std::uint64_t
device_vsm_page_map_digest(const DeviceVsmPageMap &map) noexcept {
  std::uint64_t value = 1469598103934665603ull;
  for (std::size_t index = 0u; index < map.words.size(); ++index) {
    if (index == 5u || index == 6u) {
      continue;
    }
    value ^= map.words[index];
    value *= 1099511628211ull;
  }
  return value == 0u ? 1u : value;
}

inline void device_vsm_page_map_reset(DeviceVsmPageMap &map) noexcept {
  map = {};
}

inline bool
device_vsm_page_map_store_row(DeviceVsmPageMap &map, const std::size_t index,
                              const DeviceVsmPageMapRow &row) noexcept {
  if (index >= DeviceVsmPageMapRowCapacity ||
      row.count > DeviceVsmPageMapEntryCapacity) {
    return false;
  }
  const std::size_t base = device_vsm_page_map_row_base(index);
  map.words[base] = row.resource;
  map.words[base + 1u] = row.external_slot;
  map.words[base + 2u] = row.frame_capacity;
  map.words[base + 3u] = static_cast<std::uint32_t>(row.page_bytes);
  map.words[base + 4u] = static_cast<std::uint32_t>(row.page_bytes >> 32u);
  map.words[base + 5u] = row.count;
  for (std::size_t entry = 0u; entry < DeviceVsmPageMapEntryCapacity; ++entry) {
    const std::size_t offset = base + 6u + entry * 3u;
    map.words[offset] = row.entries[entry].target;
    map.words[offset + 1u] = row.entries[entry].source;
    map.words[offset + 2u] = row.entries[entry].origin;
  }
  return true;
}

[[nodiscard]] inline bool
device_vsm_page_map_load_row(const DeviceVsmPageMap &map,
                             const std::size_t index,
                             DeviceVsmPageMapRow &row) noexcept {
  if (index >= DeviceVsmPageMapRowCapacity) {
    return false;
  }
  const std::size_t base = device_vsm_page_map_row_base(index);
  row = {};
  row.resource = map.words[base];
  row.external_slot = map.words[base + 1u];
  row.frame_capacity = map.words[base + 2u];
  row.page_bytes = static_cast<std::uint64_t>(map.words[base + 3u]) |
                   (static_cast<std::uint64_t>(map.words[base + 4u]) << 32u);
  row.count = map.words[base + 5u];
  for (std::size_t entry = 0u; entry < DeviceVsmPageMapEntryCapacity; ++entry) {
    const std::size_t offset = base + 6u + entry * 3u;
    row.entries[entry] =
        DeviceVsmPageMapEntry{.target = map.words[offset],
                              .source = map.words[offset + 1u],
                              .origin = map.words[offset + 2u]};
  }
  return true;
}

inline void device_vsm_page_map_seal(DeviceVsmPageMap &map, const bool active,
                                     const std::uint32_t row_count,
                                     const std::uint32_t total_count) noexcept {
  if (!active) {
    device_vsm_page_map_reset(map);
    return;
  }
  map.words[0u] = DeviceVsmPageMapMagic;
  map.words[1u] = DeviceVsmPageMapVersion;
  map.words[2u] = 1u;
  map.words[3u] = row_count;
  map.words[4u] = total_count;
  map.words[5u] = 0u;
  map.words[6u] = 0u;
  map.words[7u] = 0u;
  const std::uint64_t digest = device_vsm_page_map_digest(map);
  map.words[5u] = static_cast<std::uint32_t>(digest);
  map.words[6u] = static_cast<std::uint32_t>(digest >> 32u);
}

[[nodiscard]] inline bool
device_vsm_page_map_active(const DeviceVsmPageMap &map) noexcept {
  return map.words[2u] != 0u;
}

[[nodiscard]] inline bool device_vsm_page_map_valid(
    const DeviceVsmPageMap &map,
    const std::uint64_t page_count =
        std::numeric_limits<std::uint64_t>::max()) noexcept {
  if (map.words[0u] == 0u && map.words[1u] == 0u && map.words[2u] == 0u &&
      map.words[3u] == 0u && map.words[4u] == 0u && map.words[5u] == 0u &&
      map.words[6u] == 0u && map.words[7u] == 0u) {
    for (std::size_t index = DeviceVsmPageMapHeaderWords;
         index < map.words.size(); ++index) {
      if (map.words[index] != 0u) {
        return false;
      }
    }
    return true;
  }
  if (map.words[0u] != DeviceVsmPageMapMagic ||
      map.words[1u] != DeviceVsmPageMapVersion || map.words[2u] != 1u ||
      map.words[3u] == 0u || map.words[3u] > DeviceVsmPageMapRowCapacity ||
      map.words[4u] == 0u || map.words[4u] > 512u || map.words[7u] != 0u) {
    return false;
  }
  const std::uint64_t expected = device_vsm_page_map_digest(map);
  const std::uint64_t supplied =
      static_cast<std::uint64_t>(map.words[5u]) |
      (static_cast<std::uint64_t>(map.words[6u]) << 32u);
  if (expected != supplied) {
    return false;
  }
  std::array<bool, DeviceVsmPageMapRowCapacity> slots{};
  std::uint32_t total = 0u;
  for (std::size_t index = 0u; index < map.words[3u]; ++index) {
    DeviceVsmPageMapRow row{};
    if (!device_vsm_page_map_load_row(map, index, row) || row.resource == 0u ||
        row.external_slot >= DeviceVsmPageMapRowCapacity ||
        slots[row.external_slot] || row.frame_capacity == 0u ||
        row.frame_capacity > DeviceVsmPageMapEntryCapacity ||
        row.page_bytes == 0u || row.count != row.frame_capacity ||
        row.count > DeviceVsmPageMapEntryCapacity ||
        total > std::numeric_limits<std::uint32_t>::max() - row.count) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      DeviceVsmPageMapRow previous{};
      if (!device_vsm_page_map_load_row(map, prior, previous) ||
          previous.resource == row.resource) {
        return false;
      }
    }
    slots[row.external_slot] = true;
    total += row.count;
    std::array<bool, DeviceVsmPageMapEntryCapacity> targets{};
    std::array<bool, DeviceVsmPageMapEntryCapacity> sources{};
    for (std::size_t entry = 0u; entry < row.count; ++entry) {
      const DeviceVsmPageMapEntry value = row.entries[entry];
      if (value.target >= row.frame_capacity ||
          value.source >= row.frame_capacity || value.origin > 1u ||
          targets[value.target] || sources[value.source]) {
        return false;
      }
      targets[value.target] = true;
      sources[value.source] = true;
    }
    for (std::size_t entry = row.count; entry < DeviceVsmPageMapEntryCapacity;
         ++entry) {
      if (row.entries[entry].target != 0u || row.entries[entry].source != 0u ||
          row.entries[entry].origin != 0u) {
        return false;
      }
    }
    if (page_count != std::numeric_limits<std::uint64_t>::max() &&
        page_count != 0u) {
      const std::uint64_t tail = page_count % row.frame_capacity;
      if (tail != 0u) {
        std::array<bool, DeviceVsmPageMapEntryCapacity> tail_sources{};
        for (std::size_t entry = 0u; entry < row.count; ++entry) {
          const DeviceVsmPageMapEntry value = row.entries[entry];
          if (value.target < tail) {
            if (value.source >= tail || tail_sources[value.source]) {
              return false;
            }
            tail_sources[value.source] = true;
          }
        }
        for (std::size_t target = 0u; target < tail; ++target) {
          bool found = false;
          for (std::size_t entry = 0u; entry < row.count; ++entry) {
            found = found || row.entries[entry].target == target;
          }
          if (!found || !tail_sources[target]) {
            return false;
          }
        }
      }
    }
  }
  if (total != map.words[4u]) {
    return false;
  }
  for (std::size_t index = map.words[3u]; index < DeviceVsmPageMapRowCapacity;
       ++index) {
    DeviceVsmPageMapRow row{};
    if (!device_vsm_page_map_load_row(map, index, row) || row.resource != 0u ||
        row.external_slot != 0u || row.frame_capacity != 0u ||
        row.page_bytes != 0u || row.count != 0u) {
      return false;
    }
    for (const DeviceVsmPageMapEntry value : row.entries) {
      if (value.target != 0u || value.source != 0u || value.origin != 0u) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] inline bool
device_vsm_page_map_equal(const DeviceVsmPageMap &left,
                          const DeviceVsmPageMap &right) noexcept {
  return left.words == right.words;
}

} // namespace rund::node::accel::detail
