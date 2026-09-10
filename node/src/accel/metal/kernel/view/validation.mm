#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool MetalViewRequiresLowering(const BoundStep &source) noexcept {
  return source.step != nullptr && source.source_binds != nullptr &&
         source.step->kind() != rund::kernel::NodeKind::Map &&
         source.step->kind() != rund::kernel::NodeKind::ScatterReduce;
}

bool ValidateMetalViewSource(const BoundStep &source) noexcept {
  return source.source_binds != nullptr && source.source_binds->valid();
}

bool MetalViewReferenceNeedsLowering(const rund::kernel::ResidentBufferRef &ref,
                                     bool &normalize_singleton) noexcept {
  // Stride cannot change the selected address set for zero or one element.
  const bool dense = ref.count > 1u && ref.stride_bytes != ref.element_bytes;
  normalize_singleton =
      ref.count == 1u && ref.stride_bytes != ref.element_bytes;
  return (dense || normalize_singleton) && ref.element_bytes != 0u &&
         ref.count <=
             std::numeric_limits<std::uint64_t>::max() / ref.element_bytes;
}

#endif

} // namespace rund::node::accel::detail
