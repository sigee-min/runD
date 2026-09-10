#include "internal.hpp"

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

rund::AccelCheck Build(Context &context) noexcept {
  if (!metal_residency_sliding::BuildPayload(context.sequence, context.gate,
                                             context.descriptor, context.locals,
                                             context.storage.payload)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  const std::size_t authenticated_local_count =
      context.storage.payload.words[metal_residency_sliding::LocalCountWord];
  context.authenticated_locals = std::span<const std::uint32_t>{
      context.storage.payload.words.data() + MetalResidencySlidingLocalWord,
      authenticated_local_count};
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
