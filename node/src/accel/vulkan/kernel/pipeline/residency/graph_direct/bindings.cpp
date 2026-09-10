#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::graph_direct_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
GraphDirectResidentRef(const rund::kernel::ResidentBufferRef *const ref,
                       const std::shared_ptr<void> *const handle,
                       const std::uint32_t usage) noexcept {
  if (ref == nullptr || handle == nullptr || *handle == nullptr ||
      ref->id == 0u || ref->bytes == 0u || ref->count == 0u ||
      ref->element_bytes != sizeof(std::uint64_t) ||
      ref->stride_bytes < ref->element_bytes || ref->usage != usage) {
    return false;
  }
  std::uint64_t tail = 0u;
  std::uint64_t end = 0u;
  return rund::kernel::checked::mul(ref->count - 1u, ref->stride_bytes, tail) &&
         rund::kernel::checked::add(ref->offset_bytes, tail, end) &&
         rund::kernel::checked::add(end, ref->element_bytes, end) &&
         end <= ref->bytes;
}

[[nodiscard]] bool GraphDirectCanonicalResidentRef(
    const RunBinds &binds, const rund::kernel::ResidentBufferRef *const ref,
    const std::shared_ptr<void> *const handle, const std::uint64_t index,
    const std::uint32_t usage) noexcept {
  if (index >= binds.size() || !GraphDirectResidentRef(ref, handle, usage)) {
    return false;
  }
  const rund::kernel::ResidentBufferRef *const refs = binds.refs();
  const std::shared_ptr<void> *const handles = binds.handles();
  if (refs == nullptr || handles == nullptr || handles[index] == nullptr) {
    return false;
  }
  const rund::kernel::ResidentBufferRef &canonical = refs[index];
  return canonical.id == ref->id && canonical.bytes == ref->bytes &&
         canonical.offset_bytes == ref->offset_bytes &&
         canonical.element_bytes == ref->element_bytes &&
         canonical.stride_bytes == ref->stride_bytes &&
         canonical.count == ref->count && canonical.usage == ref->usage &&
         canonical.usage == usage && handles[index] == *handle;
}

#endif

} // namespace rund::node::accel::detail::graph_direct_detail
