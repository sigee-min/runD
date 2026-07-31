#pragma once

#include "model.hpp"

#include <cstddef>

namespace node_accel_contract::fusion {

[[nodiscard]] inline Inputs BuildInputs() noexcept {
  Inputs inputs{};
  inputs.host = {-20, -3, 0, 2, 9, 25, 64, 100};
  inputs.vel = {3, 1, -2, 8, 5, -7, 11, 4};
  for (std::size_t index = 0u; index < inputs.host.size(); ++index) {
    inputs.add14[index] = inputs.host[index] + 14;
    inputs.add21[index] = inputs.host[index] + 21;
    inputs.add7_vel[index] = inputs.host[index] + 7 + inputs.vel[index];
    inputs.add_vel[index] = inputs.host[index] + inputs.vel[index];
  }
  return inputs;
}

}  // namespace node_accel_contract::fusion
