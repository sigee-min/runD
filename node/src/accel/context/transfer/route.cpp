#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../../backend/resource.hpp"
#include "../../kernel/prepared/interface/api.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel {

detail::UploadRoute
detail::ProjectAccelBufferRoute(const rund::AccelContext &context,
                                const rund::AccelBuffer &buffer) noexcept {
  const std::shared_ptr<ContextToken> context_token =
      AdmitContextToken(context);
  if (context_token == nullptr) {
    return {};
  }
  const TransferAdmission admission =
      AdmitAccelBufferPrivateTransfer(context_token, buffer);
  return !admission.check.ok ? UploadRoute{}
                             : UploadRoute{.resident = buffer.resident,
                                           .handle = admission.route.handle,
                                           .bytes = admission.byte_extent};
}

detail::AccelHostView
detail::ReadAccelBuffer(const rund::AccelContext &context,
                        const rund::AccelBuffer &buffer) noexcept {
  const std::shared_ptr<ContextToken> context_token =
      AdmitContextToken(context);
  if (context_token == nullptr) {
    return {};
  }
  const TransferAdmission admission =
      AdmitAccelBufferPrivateTransfer(context_token, buffer);
  if (!admission.check.ok) {
    return {};
  }
  const BackendHostView view = ReadBackendBuffer(
      admission.pick, admission.route.ref, admission.route.handle);
  return !view || view.bytes < admission.byte_extent
             ? AccelHostView{}
             : AccelHostView{.data = view.data, .bytes = admission.byte_extent};
}

detail::AccelHostWriteView
detail::WriteAccelBuffer(const rund::AccelContext &context,
                         const rund::AccelBuffer &buffer) noexcept {
  const std::shared_ptr<ContextToken> context_token =
      AdmitContextToken(context);
  if (context_token == nullptr) {
    return {};
  }
  const TransferAdmission admission =
      AdmitAccelBufferPrivateTransfer(context_token, buffer);
  if (!admission.check.ok) {
    return {};
  }
  const BackendHostWriteView view = WriteBackendBuffer(
      admission.pick, admission.route.ref, admission.route.handle);
  return !view || view.bytes < admission.byte_extent
             ? AccelHostWriteView{}
             : AccelHostWriteView{.data = view.data,
                                  .bytes = admission.byte_extent};
}

} // namespace rund::node::accel
