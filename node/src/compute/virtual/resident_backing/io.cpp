#include "../../buffer/state.hpp"
#include "internal.hpp"

#include "../../backend.hpp"
#include "../../device/state.hpp"

#include <array>
#include <cstring>
#include <limits>

namespace rund::compute::detail {
namespace {

inline constexpr std::size_t DownloadCapacity = 64u;

[[nodiscard]] bool contains(const BufferState &buffer,
                            const std::uint64_t offset,
                            const std::size_t bytes) noexcept {
  return offset <= buffer.bytes && bytes <= buffer.bytes - offset;
}

[[nodiscard]] bool overlaps(const std::span<const std::byte> left,
                            const std::span<const std::byte> right) noexcept {
  if (left.empty() || right.empty()) {
    return false;
  }
  const std::uintptr_t left_begin =
      reinterpret_cast<std::uintptr_t>(left.data());
  const std::uintptr_t right_begin =
      reinterpret_cast<std::uintptr_t>(right.data());
  const std::uintptr_t limit = std::numeric_limits<std::uintptr_t>::max();
  if (left_begin > limit - left.size() || right_begin > limit - right.size()) {
    return true;
  }
  return left_begin < right_begin + right.size() &&
         right_begin < left_begin + left.size();
}

[[nodiscard]] Status scalar_read(VirtualBacking &backing,
                                 const std::span<const VirtualRead> ranges) {
  for (const VirtualRead range : ranges) {
    if (!range.bytes.empty() && range.bytes.data() == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    const Status status = backing.read(range.offset, range.bytes);
    if (!status) {
      return status;
    }
  }
  return Status::success();
}

[[nodiscard]] Status read_cpu(const BufferState &buffer,
                              const std::uint64_t offset,
                              const std::span<std::byte> output) noexcept {
  const CpuBufferState *const storage = cpu_buffer(buffer);
  if (storage == nullptr || storage->data == nullptr ||
      storage->bytes != buffer.bytes) {
    return Status::fail(Reason::TransferInvalid);
  }
  if (!output.empty()) {
    std::memcpy(output.data(), storage->data.get() + offset, output.size());
  }
  return Status::success();
}

[[nodiscard]] Status
write_cpu(BufferState &buffer, const std::uint64_t offset,
          const std::span<const std::byte> input) noexcept {
  CpuBufferState *const storage = cpu_buffer(buffer);
  if (storage == nullptr || storage->data == nullptr ||
      storage->bytes != buffer.bytes) {
    return Status::fail(Reason::TransferInvalid);
  }
  if (!input.empty()) {
    std::memcpy(storage->data.get() + offset, input.data(), input.size());
  }
  return Status::success();
}

} // namespace

std::uint64_t ResidentVirtualBacking::size_bytes() const noexcept {
  const std::shared_ptr<BufferState> &buffer =
      VirtualBackingAccess::resident(*this);
  return buffer == nullptr ? 0u : buffer->bytes;
}

Status
ResidentVirtualBacking::read(const std::uint64_t offset,
                             const std::span<std::byte> output) noexcept {
  const std::shared_ptr<BufferState> &buffer =
      VirtualBackingAccess::resident(*this);
  if (buffer == nullptr || buffer->device == nullptr ||
      !contains(*buffer, offset, output.size())) {
    return Status::fail(Reason::ShapeMismatch);
  }
  if (output.empty()) {
    return Status::success();
  }
  Status status = Status::fail(Reason::TransferInvalid);
  if (buffer->device->backend == Backend::Cpu) {
    status = read_cpu(*buffer, offset, output);
  } else if (buffer->device->ops != nullptr) {
    const BufferReadView view =
        buffer->device->ops->host_read == nullptr
            ? BufferReadView{}
            : buffer->device->ops->host_read(*buffer->device, *buffer);
    if (view && view.bytes == buffer->bytes) {
      std::memcpy(output.data(), view.data + offset, output.size());
      status = Status::success();
    } else if (buffer->device->ops->download_batch != nullptr &&
               buffer->device->ops->download_prefix_capacity != 0u) {
      std::uint64_t hash = 0u;
      node::accel::detail::DownloadRangeOutcome outcome{};
      const std::array requests{DownloadRequest{
          .buffer = buffer.get(),
          .data = output.data(),
          .bytes = output.size(),
          .offset = static_cast<std::size_t>(offset),
          .payload_hash = &hash,
          .outcome = &outcome,
      }};
      status =
          buffer->device->ops
              ->download_batch(*buffer->device, requests,
                               node::accel::detail::TransferAuthority::Shared)
              .status;
    } else if (buffer->device->ops->download != nullptr) {
      status = buffer->device->ops
                   ->download(*buffer->device, *buffer, output.data(),
                              output.size(), static_cast<std::size_t>(offset))
                   .status;
    }
  }
  if (status) {
    record_transfer(*buffer->device, output.size());
  }
  return status;
}

Status ResidentVirtualBacking::read_batch(
    const std::span<const VirtualRead> ranges) noexcept {
  if (ranges.empty()) {
    return Status::success();
  }
  const std::shared_ptr<BufferState> &buffer =
      VirtualBackingAccess::resident(*this);
  if (buffer == nullptr || buffer->device == nullptr ||
      buffer->device->backend == Backend::Cpu ||
      buffer->device->ops == nullptr ||
      buffer->device->ops->download_batch == nullptr ||
      buffer->device->ops->download_prefix_capacity == 0u ||
      ranges.size() > DownloadCapacity ||
      ranges.size() > buffer->device->ops->download_prefix_capacity) {
    return scalar_read(*this, ranges);
  }
  for (std::size_t index = 0u; index < ranges.size(); ++index) {
    const VirtualRead &range = ranges[index];
    if ((!range.bytes.empty() && range.bytes.data() == nullptr) ||
        !contains(*buffer, range.offset, range.bytes.size())) {
      return scalar_read(*this, ranges);
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (overlaps(std::span<const std::byte>{ranges[prior].bytes.data(),
                                              ranges[prior].bytes.size()},
                   std::span<const std::byte>{range.bytes.data(),
                                              range.bytes.size()})) {
        return scalar_read(*this, ranges);
      }
    }
  }

  std::array<DownloadRequest, DownloadCapacity> requests{};
  std::array<node::accel::detail::DownloadRangeOutcome, DownloadCapacity>
      outcomes{};
  std::array<std::uint64_t, DownloadCapacity> hashes{};
  for (std::size_t index = 0u; index < ranges.size(); ++index) {
    const VirtualRead &range = ranges[index];
    requests[index] = DownloadRequest{
        .buffer = buffer.get(),
        .data = range.bytes.data(),
        .bytes = range.bytes.size(),
        .offset = static_cast<std::size_t>(range.offset),
        .payload_hash = &hashes[index],
        .outcome = &outcomes[index],
    };
  }
  const DownloadResult result = buffer->device->ops->download_batch(
      *buffer->device,
      std::span<const DownloadRequest>{requests.data(), ranges.size()},
      node::accel::detail::TransferAuthority::Shared);
  for (std::size_t index = 0u; index < ranges.size(); ++index) {
    const node::accel::detail::DownloadRangeOutcome &outcome = outcomes[index];
    if (outcome.state != node::accel::detail::DownloadRangeState::Complete) {
      return result.status ? Status::fail(Reason::TransferInvalid)
                           : result.status;
    }
    if (outcome.confirmed_bytes != ranges[index].bytes.size()) {
      return Status::fail(Reason::TransferInvalid);
    }
    if (!ranges[index].bytes.empty()) {
      record_transfer(*buffer->device, ranges[index].bytes.size());
    }
  }
  return result.status;
}

Status
ResidentVirtualBacking::write(const std::uint64_t offset,
                              const std::span<const std::byte> input) noexcept {
  const std::shared_ptr<BufferState> &buffer =
      VirtualBackingAccess::resident(*this);
  if (buffer == nullptr || buffer->device == nullptr ||
      !contains(*buffer, offset, input.size())) {
    return Status::fail(Reason::ShapeMismatch);
  }
  if (input.empty()) {
    return Status::success();
  }
  Status status = Status::fail(Reason::TransferInvalid);
  if (buffer->device->backend == Backend::Cpu) {
    status = write_cpu(*buffer, offset, input);
  } else if (buffer->device->ops != nullptr) {
    const BufferWriteView view =
        buffer->device->ops->host_write == nullptr
            ? BufferWriteView{}
            : buffer->device->ops->host_write(*buffer->device, *buffer);
    if (view && view.bytes == buffer->bytes) {
      std::memcpy(view.data + offset, input.data(), input.size());
      status = Status::success();
    } else if (buffer->device->ops->upload_batch != nullptr) {
      const std::array requests{UploadRequest{
          .buffer = buffer.get(),
          .data = input.data(),
          .bytes = input.size(),
          .offset = static_cast<std::size_t>(offset),
      }};
      status =
          buffer->device->ops
              ->upload_batch(*buffer->device, requests,
                             node::accel::detail::TransferCompletion::Complete,
                             node::accel::detail::TransferAuthority::Shared)
              .status;
    }
  }
  if (status) {
    record_transfer(*buffer->device, input.size());
  }
  return status;
}

} // namespace rund::compute::detail
