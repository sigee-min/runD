#pragma once

#include "../../../../pipeline/residency/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::registry_model {

struct CycleSlot final {
  std::array<std::uint64_t, 2u> tokens{};
  std::array<std::uint64_t, 2u> ordinals{};
  std::array<std::uint32_t, 2u> banks{};
  std::array<std::uint64_t, 2u> terminals{NeverUse, NeverUse};
  std::size_t count{};
  std::uint64_t token{};
  bool closed{};
};

} // namespace rund::compute::detail::residency::registry_model
