#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

OpenBufferAdmission AdmitAccelBufferOpen(const rund::AccelContext &context,
                                         const rund::Buffer &buffer,
                                         const rund::AccelBufferDesc &desc) {
  return AdmitAccelBufferOpen(AdmitContextToken(context), buffer, desc);
}

OpenBufferAdmission
AdmitAccelBufferOpen(const std::shared_ptr<ContextToken> &context,
                     const rund::Buffer &buffer,
                     const rund::AccelBufferDesc &desc) {
  if (context == nullptr || !buffer.check.ok || buffer.id == 0u ||
      buffer.bytes == 0u || buffer.owner == nullptr ||
      buffer.handle == nullptr || !SameObject(context->pick, buffer.owner) ||
      !KnownUsage(buffer.usage) || !UsageCompatible(desc.usage, buffer.usage)) {
    return OpenBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const rund::AccelCheck desc_check = CheckDesc(desc);
  if (!desc_check.ok) {
    return OpenBufferAdmission{.check = desc_check};
  }

  const std::uint64_t byte_extent = desc.scalar_width_bytes * desc.count;
  const std::shared_ptr<void> handle = ResidentHandle(buffer);
  const rund::kernel::ResidentBufferRef requested{
      .id = buffer.id,
      .bytes = buffer.bytes,
      .element_bytes = buffer.element_bytes,
      .stride_bytes = buffer.stride_bytes,
      .count = buffer.count,
      .usage = ResidentUsage(buffer.usage),
  };
  const BackendLookup lookup =
      LookupBackendBuffer(context->pick, requested, handle);
  if (requested.id == 0u || handle == nullptr || !lookup.check.ok ||
      !PublicBufferMatchesCanonical(buffer, lookup.ref) ||
      !SameObject(handle, lookup.handle)) {
    return OpenBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  if (byte_extent > lookup.ref.bytes) {
    return OpenBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_overflow")};
  }

  return OpenBufferAdmission{
      .check = OkAccelCheck(),
      .context = context,
      .handle = handle,
      .resident = ResidentRefFrom(lookup.ref, desc),
      .byte_extent = byte_extent,
  };
}

} // namespace rund::node::accel::detail
