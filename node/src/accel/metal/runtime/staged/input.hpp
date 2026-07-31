#pragma once

#include "../local.hpp"

#include <cstddef>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool PrepareStagedInputBuffer(
    MetalAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings, const StagedProof &staged,
    ScopedMetalBuffers &scoped, MetalRuntimeBuffer *&input_buffer) {
  input_buffer = nullptr;
  if (plan.input_buffer_count == 0u) {
    return true;
  }
  rund::kernel::u64 input_byte_count = 0u;
  if (!StagedInputByteCount(bindings, window, 1u, input_byte_count)) {
    return false;
  }
  std::size_t input_size_bytes = 0u;
  if (!ToSize(input_byte_count, input_size_bytes)) {
    return false;
  }
  MetalRuntimeBuffer &staged_input = scoped.add(
      AcquireMetalBuffer(adapter, input_size_bytes, MetalBufferUsage::Input));
  if (staged_input.buffer == nullptr) {
    return false;
  }
  input_buffer = &staged_input;
  auto *const input_data =
      static_cast<std::byte *>(MetalBufferContents(*input_buffer));
  if (input_data == nullptr) {
    return false;
  }

  rund::kernel::u64 input_cursor = 0u;
  rund::kernel::u64 semantic_input_bytes = 0u;
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    rund::kernel::u64 input_offset = 0u;
    rund::kernel::u64 input_range = 0u;
    rund::kernel::u64 next_cursor = 0u;
    if (!StagedInputRange(bindings.input_buffers[index], window, input_cursor,
                          1u, input_offset, input_range, next_cursor)) {
      return false;
    }
    std::size_t offset_size = 0u;
    std::size_t range_size = 0u;
    if (!ToSize(input_offset, offset_size) ||
        !ToSize(input_range, range_size) || offset_size > input_size_bytes ||
        range_size > input_size_bytes - offset_size ||
        !PackInputBufferRange(bindings.input_buffers[index], bindings, window,
                              input_data + offset_size, range_size,
                              staged.bulk())) {
      return false;
    }
    if (!rund::kernel::checked::add(semantic_input_bytes, input_range)) {
      return false;
    }
    semantic_input_bytes += input_range;
    input_cursor = next_cursor;
  }
  RecordMetalHostToDeviceBytes(adapter, semantic_input_bytes);
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
