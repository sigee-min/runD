#pragma once

#include <cstdint>

namespace rund::compute {

enum class PipelineNestedPhase : std::uint8_t {
  None,
  Seed,
  Action,
  Fold,
};

} // namespace rund::compute
