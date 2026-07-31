#pragma once

#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include "../../backend/match.hpp"
#include "../../kernel/memory.hpp"
#include "../object.hpp"
#include "../state.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline MetalAdapter *
MetalAdapterFromPick(const rund::AccelDevice &pick) noexcept {
  if (!MetalPickOwnsAdapter(pick)) {
    return nullptr;
  }
  return static_cast<MetalAdapter *>(pick.backend.context);
}

[[nodiscard]] inline PreparedMemory
MetalBufferMemory(const MetalRuntimeBuffer &buffer,
                  const std::uint64_t budget) noexcept {
  return PreparedMemory{.current = buffer.bytes,
                        .peak = buffer.bytes,
                        .cumulative = buffer.bytes,
                        .reused = buffer.reused ? buffer.bytes : 0u,
                        .budget = budget};
}

template <class... Buffer>
[[nodiscard]] inline PreparedMemory
MetalBuffersMemory(const std::uint64_t budget,
                   const Buffer &...buffers) noexcept {
  PreparedMemory memory{};
  (accumulate_memory(memory, MetalBufferMemory(buffers, budget)), ...);
  return memory;
}

} // namespace rund::node::accel::detail
