#include "../resource.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

PreparedMemory MetalNumericMemory(const std::shared_ptr<void> &prepared,
                                  const std::uint64_t budget) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const state =
      static_cast<const MetalNumericPrepared *>(prepared.get());
  return state == nullptr ? PreparedMemory{}
                          : MetalBufferMemory(state->twiddle, budget);
#else
  (void)prepared;
  (void)budget;
  return {};
#endif
}

bool DescribeMetalNumericPipelineStatus(
    const std::shared_ptr<void> &prepared,
    MetalPipelineStatusBindings &bindings) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  bindings = {};
  const auto *const state =
      static_cast<const MetalNumericPrepared *>(prepared.get());
  if (state == nullptr || !state->semantic_status ||
      state->status_index >= state->buffer_count || state->status_count == 0u ||
      state->status_count > std::numeric_limits<std::uint32_t>::max() ||
      state->status_count >
          std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t) ||
      state->buffers[state->status_index].device_buffer == nullptr) {
    return false;
  }
  std::array<std::uint32_t, 4u> reasons{};
  for (std::uint32_t raw = 1u; raw <= reasons.size(); ++raw) {
    reasons[raw - 1u] = static_cast<std::uint32_t>(
        rund::compute::detail::primitive_execution_reason(
            state->status_primitive, raw));
  }
  bindings.values[0] = MetalPipelineStatusBinding{
      .buffer = state->buffers[state->status_index].device_buffer.get(),
      .offset = state->buffers[state->status_index].ref.offset_bytes,
      .bytes = state->status_count * sizeof(std::uint32_t),
      .reasons = reasons,
      .observed_count = static_cast<std::uint32_t>(state->status_count),
      .encoding = MetalPipelineStatusEncoding::Mapping,
      .replace = false,
  };
  bindings.size = 1u;
  return true;
#else
  (void)prepared;
  (void)bindings;
  return false;
#endif
}

rund::AccelCheck FinishMetalNumeric(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &prepared) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const state = static_cast<MetalNumericPrepared *>(prepared.get());
  if (state == nullptr || state->adapter != &adapter) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  return state->status_index < state->buffer_count
             ? StatusCheck(adapter, state->buffers[state->status_index],
                           state->status_count, state->dispatches)
             : (RecordMetalDispatches(adapter, state->dispatches),
                SetMetalLastError(adapter, "ok"), rund::AccelCheck{true, "ok"});
#else
  (void)adapter;
  (void)prepared;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
