#pragma once

#include <rund/compute/abi/model.hpp>

#include "../value/arena.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rund::compute::detail {

inline constexpr std::uint32_t NoIndex = ~std::uint32_t{0u};

enum class MapReadMode : std::uint8_t {
  Element,
  Uniform,
  Indexed,
};

struct MapRead final {
  MapReadMode mode = MapReadMode::Element;
  std::uint32_t index = NoIndex;
  std::uint32_t count = 0u;

  [[nodiscard]] constexpr bool indexed() const noexcept {
    return mode == MapReadMode::Indexed;
  }

  [[nodiscard]] constexpr bool uniform() const noexcept {
    return mode == MapReadMode::Uniform;
  }
};

struct MapStep final {
  std::string name{};
  ValueIdRange inputs{};
  ValueIdRange outputs{};
  std::vector<ExprRef> expressions{};
  std::vector<MapRead> reads{};
  FlowControl control{};
};

} // namespace rund::compute::detail
