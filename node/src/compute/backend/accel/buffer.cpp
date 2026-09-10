#include "../../device/state.hpp"
#include "../../buffer/state.hpp"
#include "local.hpp"

#include "../../../accel/backend/token.hpp"
#include "../../../accel/context/local.hpp"
#include "../../status.hpp"
#include "../../type.hpp"

#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>

#include <utility>

namespace rund::compute::detail::accel_backend {

Status project_buffer_view(const BufferState &owner, BufferState &view) {
  const AccelDeviceState *const device =
      owner.device == nullptr ? nullptr : accel_device(*owner.device);
  const AccelBufferState *const source = accel_buffer(owner);
  if (device == nullptr || source == nullptr || view.device != owner.device ||
      view.bytes != owner.bytes ||
      view.physical_bytes != owner.physical_bytes) {
    return Status::fail(Reason::BufferCapacity);
  }
  auto projected = node::accel::detail::ProjectAccelBufferView(
      device->context, source->buffer,
      rund::AccelBufferDesc{.scalar_width_bytes = type_bytes(view.type),
                            .count = view.count,
                            .usage = source->buffer.usage});
  if (!projected) {
    return Status::fail(
        project_reason(projected.reason, Reason::BufferCapacity));
  }
  view.storage.emplace<AccelBufferState>(
      AccelBufferState{.buffer = std::move(projected)});
  return Status::success();
}

Status allocate_buffer(DeviceState &device, BufferState &buffer,
                       const std::size_t scalar_bytes, const std::size_t count,
                       const bool zero_initialize,
                       const node::accel::detail::BackendBufferMemory memory,
                       const std::uint64_t exact_storage_bytes) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  rund::AccelBuffer created =
      node::accel::detail::CreateAccelBufferWithInitialization(
          accel->context,
          rund::AccelBufferDesc{
              .scalar_width_bytes = scalar_bytes,
              .count = count,
              .usage = rund::BufferUsage::ReadWrite,
          },
          zero_initialize
              ? node::accel::detail::BackendBufferInitialization::Zeroed
              : node::accel::detail::BackendBufferInitialization::FullOverwrite,
          memory, exact_storage_bytes);
  if (!created.check.ok) {
    return Status::fail(
        project_reason(created.check.reason, Reason::BufferCapacity));
  }
  buffer.storage.emplace<AccelBufferState>(
      AccelBufferState{.buffer = std::move(created)});
  const AccelBufferState *const stored = accel_buffer(buffer);
  buffer.physical_bytes =
      stored == nullptr || stored->buffer.buffer.storage_bytes == 0u
          ? buffer.bytes
          : stored->buffer.buffer.storage_bytes;
  return Status::success();
}

std::uint64_t buffer_storage_bytes(const DeviceState &device,
                                   const std::uint64_t logical_bytes) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return 0u;
  }
  const auto token = node::accel::detail::AdmitPick(accel->pick);
  return token == nullptr || token->ops == nullptr ||
                 token->ops->buffer_storage_bytes == nullptr
             ? 0u
             : token->ops->buffer_storage_bytes(token->raw, logical_bytes);
}

std::uint64_t
pipeline_transfer_storage_bytes(const DeviceState &device,
                                const std::uint64_t logical_bytes) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return 0u;
  }
  const auto token = node::accel::detail::AdmitPick(accel->pick);
  return token == nullptr || token->ops == nullptr ||
                 token->ops->pipeline_transfer_storage_bytes == nullptr
             ? 0u
             : token->ops->pipeline_transfer_storage_bytes(token->raw,
                                                           logical_bytes);
}

} // namespace rund::compute::detail::accel_backend
