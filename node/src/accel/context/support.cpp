#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "admission/local.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

ContextAdmission AdmitContextForSupport(const rund::AccelContext &context) {
  const ContextTokenAdmission admission = AdmitContextToken(context);
  return SupportAdmissionFrom(admission);
}

rund::AccelCheck
ValidateAccelBufferForSupport(const rund::AccelContext &context,
                              const rund::AccelBuffer &buffer) {
  return AdmitAccelBufferTransfer(context, buffer).check;
}

rund::AccelCheck ValidateAccelBufferForSupport(
  const rund::AccelContext &context, const ContextAdmission &admission,
  const rund::AccelBuffer &buffer) {
  if (!ContextMatchesAdmission(context, admission)) {
    return RejectAccelCheck("accel_context_buffer_invalid");
  }
  return AdmitAccelBufferForSupport(admission, buffer).check;
}

rund::AccelCheck ValidateAccelBufferForSupport(
    const ContextAdmission &admission, const rund::AccelBuffer &buffer) {
  return AdmitAccelBufferForSupport(admission, buffer).check;
}

rund::AccelCheck ValidateAccelBufferForSupport(
    const ContextAdmission &admission, const rund::AccelBuffer &buffer,
    std::shared_ptr<void> &backend_handle) {
  const SupportBufferAdmission support =
      AdmitAccelBufferForSupport(admission, buffer);
  if (!support.check.ok) {
    backend_handle.reset();
    return support.check;
  }
  backend_handle = support.lookup.handle;
  return support.check;
}

} // namespace rund::node::accel::detail
