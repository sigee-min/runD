#include "record.hpp"

#include "../../../runtime/replay/host/payload/diagnostic/ring.hpp"
#include "../../../runtime/task/scheduler/host.hpp"
#include "../native/result.hpp"
#include "../scheduler.hpp"

#include <span>

namespace rund::net {
namespace {

[[nodiscard]] ::rund::host::Event
Event(const NetEventRequest request) noexcept {
  return ::rund::host::Event{
      .kind = request.kind,
      .status = StatusForNative(request.native),
      .host_handle_id = request.socket_id,
      .requested_bytes = request.requested_bytes,
      .completed_bytes =
          request.native.value < 0
              ? 0u
              : CompletedByteCount(request.native, request.requested_bytes),
      .native_errno = request.native.error,
      .name_hash = request.name_hash,
      .payload_hash = request.payload_hash,
  };
}

[[nodiscard]] std::span<const std::byte>
ContiguousSlice(const void *const context, const std::size_t) noexcept {
  return *static_cast<const std::span<const std::byte> *>(context);
}

[[nodiscard]] std::span<const std::byte>
VectoredSlice(const void *const context, const std::size_t index) noexcept {
  const auto slices =
      *static_cast<const std::span<const batch::Buffer> *>(context);
  if (index >= slices.size()) {
    return {};
  }
  return {slices[index].data, slices[index].size};
}

} // namespace

bool RecordNetEvent(const NetEventRequest request) noexcept {
  return RecordHostEvent(Event(request));
}

bool RecordNetIngressEvent(const NetEventRequest request,
                           const std::span<const std::byte> bytes) noexcept {
  ::rund::host::Event event = Event(request);
  const node::replay_detail::payload::RawByteSource source{
      .context = &bytes,
      .slice_count = 1u,
      .admitted_bytes = request.requested_bytes,
      .byte_count = event.completed_bytes,
      .slice = ContiguousSlice,
  };
  if (event.status == ::rund::host::Status::Ok &&
      !node::scheduler_host::CapturesIngress()) {
    event.payload_hash = node::replay_detail::payload::HashIngress(source);
  }
  return node::scheduler_host::Record(event, source);
}

bool RecordNetIngressEvent(
    const NetEventRequest request,
    const std::span<const batch::Buffer> slices) noexcept {
  ::rund::host::Event event = Event(request);
  const node::replay_detail::payload::RawByteSource source{
      .context = &slices,
      .slice_count = slices.size(),
      .admitted_bytes = request.requested_bytes,
      .byte_count = event.completed_bytes,
      .slice = VectoredSlice,
  };
  if (event.status == ::rund::host::Status::Ok &&
      !node::scheduler_host::CapturesIngress()) {
    event.payload_hash = node::replay_detail::payload::HashIngress(source);
  }
  return node::scheduler_host::Record(event, source);
}

} // namespace rund::net
