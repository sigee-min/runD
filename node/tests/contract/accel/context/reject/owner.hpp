#pragma once

#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "local.hpp"

namespace node_accel_contract::context::reject {

[[nodiscard]] inline bool RejectsOwnerTransplants(const State &state,
                                                  TransferFixture &io) {
  rund::AccelContext transplanted_context_owner = state.second;
  transplanted_context_owner.owner = state.context.owner;
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       transplanted_context_owner, state.created,
                       io.upload.data(), sizeof(io.upload)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  if (!CheckReason(
          rund::node::accel::UploadAccelBuffer(
              state.second, state.created, io.upload.data(), sizeof(io.upload)),
          "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer transplanted_buffer_owner = state.created;
  transplanted_buffer_owner.owner = state.second.owner;
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       state.context, transplanted_buffer_owner,
                       io.upload.data(), sizeof(io.upload)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer transplanted_route_owner = state.created;
  transplanted_route_owner.buffer.owner = state.second.owner;
  if (!CheckReason(rund::node::accel::DownloadAccelBuffer(
                       state.context, transplanted_route_owner,
                       io.download.data(), sizeof(io.download)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer transplanted_context_id = state.created;
  transplanted_context_id.context_id = state.second.id;
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       state.context, transplanted_context_id, io.upload.data(),
                       sizeof(io.upload)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer fully_transplanted = state.created;
  fully_transplanted.context_id = state.second.id;
  fully_transplanted.owner = state.second.owner;
  fully_transplanted.buffer.owner = state.second.owner;
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       state.second, fully_transplanted, io.upload.data(),
                       sizeof(io.upload)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer wrong_context_owner = state.created;
  wrong_context_owner.owner = std::make_shared<int>(17);
  if (!CheckReason(rund::node::accel::DownloadAccelBuffer(
                       state.context, wrong_context_owner, io.download.data(),
                       sizeof(io.download)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer wrong_capability = state.created;
  wrong_capability.handle = std::make_shared<int>(19);
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       state.context, wrong_capability, io.upload.data(),
                       sizeof(io.upload)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer alias_capability = state.created;
  alias_capability.handle =
      std::shared_ptr<void>(state.created.handle.get(), [](void *) {});
  if (!CheckReason(rund::node::accel::DownloadAccelBuffer(
                       state.context, alias_capability, io.download.data(),
                       sizeof(io.download)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  int alias_value{};
  alias_capability = state.created;
  alias_capability.handle =
      std::shared_ptr<void>(state.created.handle, &alias_value);
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       state.context, alias_capability, io.upload.data(),
                       sizeof(io.upload)),
                   "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBuffer alias_context_owner = state.created;
  alias_context_owner.owner =
      std::shared_ptr<void>(state.created.owner.get(), [](void *) {});
  return CheckReason(
      rund::node::accel::UploadAccelBuffer(state.context, alias_context_owner,
                                           io.upload.data(), sizeof(io.upload)),
      "accel_context_buffer_invalid");
}

} // namespace node_accel_contract::context::reject
