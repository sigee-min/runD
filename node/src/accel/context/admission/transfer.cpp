#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "local.hpp"

#include <utility>

namespace rund::node::accel::detail {

TransferAdmission AdmitAccelBufferTransfer(const rund::AccelContext &context,
                                           const rund::AccelBuffer &buffer) {
  return AdmitAccelBufferTransfer(AdmitContextToken(context), buffer);
}

TransferAdmission
AdmitAccelBufferTransfer(const std::shared_ptr<ContextToken> &context,
                         const rund::AccelBuffer &buffer) {
  SupportBufferAdmission support =
      AdmitAccelBufferForSupport(SupportAdmissionFrom(context), buffer);
  if (!support.check.ok) {
    return TransferAdmission{.check = support.check};
  }

  return TransferAdmission{
      .check = OkAccelCheck(),
      .pick = context->pick,
      .route = std::move(support.lookup),
      .byte_extent = support.byte_extent,
  };
}

TransferAdmission
AdmitAccelBufferPrivateTransfer(const std::shared_ptr<ContextToken> &context,
                                const rund::AccelBuffer &buffer) noexcept {
  const ContextAdmission admission = SupportAdmissionFrom(context);
  const std::shared_ptr<AccelBufferToken> token =
      LookupAccelBufferToken(buffer.handle);
  if (!AccelBufferTokenMatches(admission, buffer, token) ||
      !KnownUsage(buffer.buffer.usage) ||
      !UsageCompatible(buffer.usage, buffer.buffer.usage)) {
    return TransferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  const rund::AccelBufferDesc desc{
      .scalar_width_bytes = buffer.scalar_width_bytes,
      .count = buffer.count,
      .usage = buffer.usage,
  };
  const rund::AccelCheck desc_check = CheckDesc(desc);
  if (!desc_check.ok) {
    return TransferAdmission{.check = desc_check};
  }
  const rund::kernel::ResidentBufferRef canonical{
      .id = token->backend_id,
      .bytes = token->backend_bytes,
      .element_bytes = token->backend_element_bytes,
      .stride_bytes = token->backend_stride_bytes,
      .count = token->backend_count,
      .usage = ResidentUsage(token->backend_usage),
  };
  const std::uint64_t byte_extent = desc.scalar_width_bytes * desc.count;
  if (!PublicBufferMatchesCanonical(buffer.buffer, canonical) ||
      byte_extent != buffer.byte_extent || byte_extent > canonical.bytes ||
      !SameResidentRef(buffer.resident, ResidentRefFrom(canonical, desc)) ||
      token->backend_handle == nullptr) {
    return TransferAdmission{
        .check = RejectAccelCheck("accel_context_buffer_invalid")};
  }
  return TransferAdmission{
      .check = OkAccelCheck(),
      .pick = context->pick,
      .route = BackendLookup{.check = OkAccelCheck(),
                             .ref = canonical,
                             .handle = token->backend_handle},
      .byte_extent = byte_extent,
  };
}

rund::AccelCheck TransferCheckFrom(const rund::AccelCheck check,
                                   const char *const overflow_reason) noexcept {
  if (check.ok) {
    return OkAccelCheck();
  }
  if (SameReason(check.reason, overflow_reason)) {
    return RejectAccelCheck(overflow_reason);
  }
  // Admission already authenticated the context, owner, shape, and range.
  // Replacing a backend execution failure here destroys the first causal
  // reason (notably compute_device_lost during snapshot readback).
  return check;
}

} // namespace rund::node::accel::detail
