#pragma once

#include "state.hpp"

namespace node_accel_contract::context {

[[nodiscard]] State OpenState();

}  // namespace node_accel_contract::context

namespace node_accel_contract {

[[nodiscard]] bool ContextRejectsInvalidInputs();
[[nodiscard]] bool PublicContextApiOpensRealBackend();
[[nodiscard]] bool PublicContextApiCreatesAndTransfers();
[[nodiscard]] bool PublicContextApiRejectsForgedBuffers();

}  // namespace node_accel_contract
