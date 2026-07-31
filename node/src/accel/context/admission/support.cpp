#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

ContextAdmission SupportAdmissionFrom(const ContextTokenAdmission &admission) {
  if (!admission.check.ok) {
    return ContextAdmission{.check = admission.check};
  }
  return ContextAdmission{
      .check = admission.check,
      .context_id = admission.token->id,
      .api = admission.token->api,
      .caps = admission.token->caps,
      .owner = PublicTokenOwner(admission.token),
      .pick = admission.token->pick,
  };
}

SupportBufferAdmission
AdmitAccelBufferForSupport(const ContextAdmission &admission,
                           const rund::AccelBuffer &buffer) {
  const rund::AccelBufferDesc desc{
      .scalar_width_bytes = buffer.scalar_width_bytes,
      .count = buffer.count,
      .usage = buffer.usage,
  };
  const rund::AccelCheck desc_check = CheckDesc(desc);
  const std::shared_ptr<AccelBufferToken> token =
      LookupAccelBufferToken(buffer.handle);
  if (!AccelBufferTokenMatches(admission, buffer, token) ||
      !KnownUsage(buffer.buffer.usage) ||
      !UsageCompatible(buffer.usage, buffer.buffer.usage)) {
    return SupportBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  if (!desc_check.ok) {
    return SupportBufferAdmission{.check = desc_check};
  }

  const std::uint64_t byte_extent = desc.scalar_width_bytes * desc.count;
  if (buffer.byte_extent != byte_extent) {
    return SupportBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }

  const rund::kernel::ResidentBufferRef requested{
      .id = buffer.buffer.id,
      .bytes = buffer.buffer.bytes,
      .element_bytes = buffer.buffer.element_bytes,
      .stride_bytes = buffer.buffer.stride_bytes,
      .count = buffer.buffer.count,
      .usage = ResidentUsage(buffer.buffer.usage),
  };
  const BackendLookup lookup =
      LookupBackendBuffer(admission.pick, requested, token->backend_handle);
  if (!lookup.check.ok ||
      !PublicBufferMatchesCanonical(buffer.buffer, lookup.ref) ||
      !SameObject(token->backend_handle, lookup.handle)) {
    return SupportBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  if (byte_extent > lookup.ref.bytes) {
    return SupportBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_overflow")};
  }

  const rund::kernel::ResidentBufferRef resident =
      ResidentRefFrom(lookup.ref, desc);
  if (!SameResidentRef(buffer.resident, resident)) {
    return SupportBufferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  return SupportBufferAdmission{
      .check = OkAccelCheck(),
      .lookup = lookup,
      .byte_extent = byte_extent,
  };
}

} // namespace rund::node::accel::detail
