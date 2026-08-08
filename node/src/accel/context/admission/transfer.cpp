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
