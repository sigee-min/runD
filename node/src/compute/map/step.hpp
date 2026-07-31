#pragma once

#include <rund/compute/abi/model.hpp>

#include "../value/arena.hpp"

#include <string>
#include <vector>

namespace rund::compute::detail {

inline constexpr std::uint32_t NoIndex = ~std::uint32_t{0u};

struct MapRead final {
  std::uint32_t index = NoIndex;
  std::uint32_t count = 0u;

  [[nodiscard]] constexpr bool indexed() const noexcept {
    return index != NoIndex;
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
