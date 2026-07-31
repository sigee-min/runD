#pragma once

#include "refs.hpp"

namespace rund::node::accel::detail {

struct StepBinds {
  StepViews inputs{};
  StepViews outputs{};

  [[nodiscard]] bool valid() const noexcept {
    return inputs.valid() && outputs.valid() && outputs.size() != 0u;
  }
};

} // namespace rund::node::accel::detail
