#pragma once

#include <accel/buffer.hpp>

#include <node/accel/context.hpp>

#include "local.hpp"

namespace node_accel_contract::context::reject {

[[nodiscard]] inline bool RejectsForgedShapeFields(const State &state) {
  rund::Buffer forged_bytes = state.buffer;
  --forged_bytes.bytes;
  if (!OpenRejects(state, forged_bytes)) {
    return false;
  }

  rund::Buffer forged_width = state.buffer;
  ++forged_width.element_bytes;
  if (!OpenRejects(state, forged_width)) {
    return false;
  }

  rund::Buffer forged_count = state.buffer;
  --forged_count.count;
  if (!OpenRejects(state, forged_count)) {
    return false;
  }

  return CheckReason(
      rund::node::accel::OpenAccelBuffer(
          state.context, SyntheticBufferFor(state.context), state.typed_desc)
          .check,
      "accel_context_buffer_invalid");
}

[[nodiscard]] inline bool RejectsForgedIdentityFields(const State &state) {
  rund::Buffer forged_id = state.buffer;
  ++forged_id.id;
  if (!OpenRejects(state, forged_id)) {
    return false;
  }

  rund::Buffer forged_handle = state.buffer;
  forged_handle.handle = std::make_shared<int>(11);
  if (!OpenRejects(state, forged_handle)) {
    return false;
  }

  rund::Buffer alias_handle = state.buffer;
  alias_handle.handle =
      std::shared_ptr<void>(state.buffer.handle.get(), [](void *) {});
  if (!OpenRejects(state, alias_handle)) {
    return false;
  }

  int alias_value{};
  alias_handle = state.buffer;
  alias_handle.handle =
      std::shared_ptr<void>(state.buffer.handle, &alias_value);
  if (!OpenRejects(state, alias_handle)) {
    return false;
  }

  rund::Buffer alias_owner = state.buffer;
  alias_owner.owner =
      std::shared_ptr<void>(state.buffer.owner.get(), [](void *) {});
  if (!OpenRejects(state, alias_owner)) {
    return false;
  }

  alias_owner = state.buffer;
  alias_owner.owner = std::shared_ptr<void>(state.buffer.owner, &alias_value);
  if (!OpenRejects(state, alias_owner)) {
    return false;
  }

  rund::Buffer wrong_owner = state.buffer;
  wrong_owner.owner = std::make_shared<int>(9);
  return OpenRejects(state, wrong_owner);
}

[[nodiscard]] inline bool RejectsForgedBufferFields(const State &state) {
  return RejectsForgedShapeFields(state) && RejectsForgedIdentityFields(state);
}

} // namespace node_accel_contract::context::reject
