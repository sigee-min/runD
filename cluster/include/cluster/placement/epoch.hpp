#pragma once

#include <cstdint>

namespace rund::cluster {

struct PlacementEpoch {
  std::uint64_t value = 0u;
  friend constexpr bool operator==(const PlacementEpoch &,
                                   const PlacementEpoch &) = default;
};

} // namespace rund::cluster
