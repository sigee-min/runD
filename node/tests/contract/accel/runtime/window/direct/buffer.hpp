#pragma once

#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include "work.hpp"

#include <memory>

namespace node_accel_contract::runtime::window::direct {

struct Buffers {
  rund::Buffer lhs{};
  rund::Buffer rhs{};
  rund::Buffer output{};
  std::array<rund::kernel::ResidentBufferRef, 2u> inputs{};
  rund::kernel::ResidentBufferRef output_ref{};
  std::array<std::shared_ptr<void>, 2u> input_handles{};
  std::shared_ptr<void> output_handle{};
  rund::kernel::BindingSet bindings{};
};

[[nodiscard]] inline bool PrepareBuffers(const rund::AccelDevice &pick,
                                         const rund::compute_dsl::ComputeOp &op,
                                         const Work &work, Buffers &buffers) {
  rund::BufferDesc read_desc{
      .bytes = sizeof(work.lhs),
      .usage = rund::BufferUsage::ReadOnly,
      .alignment = 16u,
  };
  rund::BufferDesc write_desc = read_desc;
  write_desc.usage = rund::BufferUsage::WriteOnly;
  buffers.lhs = rund::node::accel::CreateBuffer(pick, read_desc);
  buffers.rhs = rund::node::accel::CreateBuffer(pick, read_desc);
  buffers.output = rund::node::accel::CreateBuffer(pick, write_desc);
  if (!buffers.lhs.check.ok || !buffers.rhs.check.ok ||
      !buffers.output.check.ok ||
      !rund::node::accel::UploadBuffer(pick, buffers.lhs, work.lhs.data(),
                                       sizeof(work.lhs))
           .ok ||
      !rund::node::accel::UploadBuffer(pick, buffers.rhs, work.rhs.data(),
                                       sizeof(work.rhs))
           .ok) {
    return false;
  }

  buffers.inputs = {
      rund::node::accel::ResidentRead(buffers.lhs, sizeof(rund::kernel::u32),
                                      kTileCount),
      rund::node::accel::ResidentRead(buffers.rhs, sizeof(rund::kernel::u32),
                                      kTileCount),
  };
  buffers.output_ref = rund::node::accel::ResidentWrite(
      buffers.output, sizeof(rund::kernel::u32), kTileCount);
  buffers.input_handles = {rund::node::accel::ResidentHandle(buffers.lhs),
                           rund::node::accel::ResidentHandle(buffers.rhs)};
  buffers.output_handle = rund::node::accel::ResidentHandle(buffers.output);
  buffers.bindings = op.resident_bindings<ResidentTileValue>(
      work.phase.phase_id, 3u, work.plan.api, buffers.inputs.data(),
      buffers.inputs.size(), buffers.output_ref, buffers.input_handles.data(),
      buffers.input_handles.size(), &buffers.output_handle);
  return buffers.bindings.ok && buffers.bindings.sequence_tiles == nullptr &&
         buffers.bindings.sequence_tile_count == 0u;
}

} // namespace node_accel_contract::runtime::window::direct
