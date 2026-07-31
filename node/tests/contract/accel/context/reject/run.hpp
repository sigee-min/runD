#pragma once

#include "desc.hpp"
#include "fields.hpp"
#include "owner.hpp"
#include "range.hpp"

namespace node_accel_contract {

[[nodiscard]] bool PublicContextApiRejectsForgedBuffers() {
  namespace r = node_accel_contract::context::reject;
  const context::State state = context::OpenState();
  if (!state.available) { return state.unavailable_ok; }

  r::TransferFixture io{};
  return r::RejectsOwnerTransplants(state, io) &&
         r::RejectsTransferRangeOverflow(state, io) &&
         r::RejectsForgedBufferFields(state) &&
         r::RejectsForgedTypedDescriptors(state);
}

}  // namespace node_accel_contract
