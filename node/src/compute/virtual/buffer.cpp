#include "state.hpp"

#include "../type.hpp"

#include <kernel/core/checked.hpp>

#include <memory>
#include <new>
#include <utility>

namespace rund::compute::detail {

Result<std::shared_ptr<VirtualBufferState>>
make_virtual_buffer(const std::uint64_t count,
                    const std::uint64_t element_bytes, const Type type,
                    const FixedFormat format,
                    std::shared_ptr<VirtualBacking> backing) noexcept {
  std::uint64_t bytes = 0u;
  if (backing == nullptr || !valid_type(type) || element_bytes == 0u ||
      element_bytes != type_bytes(type) ||
      !kernel::checked::mul(count, element_bytes, bytes)) {
    return Result<std::shared_ptr<VirtualBufferState>>::fail(
        Reason::BufferCapacity);
  }
  if (backing->size_bytes() != bytes) {
    return Result<std::shared_ptr<VirtualBufferState>>::fail(
        Reason::ShapeMismatch);
  }
  try {
    auto state = std::make_shared<VirtualBufferState>();
    state->backing = std::move(backing);
    state->type = type;
    state->format = format;
    state->count = count;
    state->element_bytes = element_bytes;
    state->bytes = bytes;
    return Result<std::shared_ptr<VirtualBufferState>>::success(
        std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<VirtualBufferState>>::fail(
        Reason::BufferCapacity);
  }
}

bool valid_virtual_buffer(
    const std::shared_ptr<VirtualBufferState> &state) noexcept {
  return state != nullptr && state->backing != nullptr &&
         state->element_bytes != 0u &&
         state->backing->size_bytes() == state->bytes;
}

std::uint64_t
virtual_buffer_size(const std::shared_ptr<VirtualBufferState> &state) noexcept {
  return state == nullptr ? 0u : state->count;
}

} // namespace rund::compute::detail
