#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include "../backend/resource.hpp"
#include "local.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel {
namespace {

[[nodiscard]] rund::AccelBuffer
SealAccelBuffer(const detail::OpenBufferAdmission &admission,
                const rund::Buffer &backend,
                const rund::AccelBufferDesc &desc) {
  const std::shared_ptr<detail::AccelBufferToken> token =
      detail::MakeAccelBufferToken(admission.context, backend,
                                   admission.resident, admission.byte_extent,
                                   desc);
  if (token == nullptr) {
    return detail::RejectBuffer(desc, backend, "accel_context_buffer_invalid");
  }
  const std::shared_ptr<void> context_owner =
      detail::PublicTokenOwner(admission.context);
  const std::shared_ptr<void> capability =
      detail::PublicBufferTokenOwner(token);
  rund::Buffer public_buffer = backend;
  public_buffer.owner = context_owner;
  public_buffer.handle = capability;
  return rund::AccelBuffer{
      .check = rund::AccelCheck{true, "ok"},
      .context_id = admission.context->id,
      .buffer = std::move(public_buffer),
      .resident = admission.resident,
      .byte_extent = admission.byte_extent,
      .scalar_width_bytes = desc.scalar_width_bytes,
      .count = desc.count,
      .usage = desc.usage,
      .owner = context_owner,
      .handle = capability,
      .reason = "ok",
  };
}

} // namespace

rund::AccelBuffer OpenAccelBuffer(const rund::AccelContext &context,
                                  const rund::Buffer &buffer,
                                  const rund::AccelBufferDesc desc) {
  const detail::OpenBufferAdmission admission =
      detail::AdmitAccelBufferOpen(context, buffer, desc);
  if (!admission.check.ok) {
    return detail::RejectBuffer(desc, buffer, admission.check.reason);
  }
  return SealAccelBuffer(admission, buffer, desc);
}

rund::AccelBuffer detail::CreateAccelBufferWithInitialization(
    const rund::AccelContext &context, const rund::AccelBufferDesc desc,
    const detail::BackendBufferInitialization initialization) {
  const rund::AccelCheck desc_check = detail::CheckDesc(desc);
  if (!desc_check.ok) {
    return detail::RejectBuffer(desc, rund::Buffer{}, desc_check.reason);
  }
  const std::shared_ptr<detail::ContextToken> context_token =
      detail::AdmitContextToken(context);
  if (context_token == nullptr) {
    return detail::RejectBuffer(desc, rund::Buffer{},
                                "accel_context_buffer_invalid");
  }

  const std::uint64_t byte_extent = desc.scalar_width_bytes * desc.count;
  const rund::Buffer buffer =
      detail::CreateBackendBuffer(context_token->pick,
                                  rund::BufferDesc{
                                      .bytes = byte_extent,
                                      .usage = desc.usage,
                                      .alignment = 16u,
                                  },
                                  initialization);
  if (!buffer.check.ok) {
    return detail::RejectBuffer(desc, buffer, buffer.check.reason);
  }

  const detail::OpenBufferAdmission opened =
      detail::AdmitAccelBufferOpen(context_token, buffer, desc);
  if (!opened.check.ok) {
    return detail::RejectBuffer(desc, buffer, opened.check.reason);
  }
  return SealAccelBuffer(opened, buffer, desc);
}

rund::AccelBuffer CreateAccelBuffer(const rund::AccelContext &context,
                                    const rund::AccelBufferDesc desc) {
  return detail::CreateAccelBufferWithInitialization(
      context, desc, detail::BackendBufferInitialization::Zeroed);
}

} // namespace rund::node::accel
