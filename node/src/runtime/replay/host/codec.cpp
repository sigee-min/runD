#include "codec.hpp"

#include <node/runtime/replay/host.hpp>

#include "../../../host/replay/layout.hpp"
#include "../artifact/bytes.hpp"
#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund::node {
namespace {

enum EventField : std::uint16_t {
  Task = 1u << 0u,
  LogicalTimeDelta = 1u << 1u,
  Stream = 1u << 2u,
  Draw = 1u << 3u,
  Handle = 1u << 4u,
  Offset = 1u << 5u,
  Requested = 1u << 6u,
  Completed = 1u << 7u,
  NativeErrno = 1u << 8u,
  NameHash = 1u << 9u,
  PathHash = 1u << 10u,
  PayloadHash = 1u << 11u,
};

inline constexpr std::uint16_t kEventFieldMask =
    Task | LogicalTimeDelta | Stream | Draw | Handle | Offset | Requested |
    Completed | NativeErrno | NameHash | PathHash | PayloadHash;

[[nodiscard]] constexpr std::uint16_t
event_field(const bool present, const EventField field) noexcept {
  return present ? static_cast<std::uint16_t>(field) : std::uint16_t{0u};
}

[[nodiscard]] bool KnownKind(const std::uint64_t value) noexcept {
  using Value = std::underlying_type_t<::rund::host::EventKind>;
  return value <=
             static_cast<std::uint64_t>(std::numeric_limits<Value>::max()) &&
         host_detail::known_replay_kind(
             static_cast<::rund::host::EventKind>(value));
}

[[nodiscard]] bool KnownStatus(const std::uint64_t value) noexcept {
  if (value > static_cast<std::uint64_t>(::rund::host::Status::Unsupported)) {
    return false;
  }
  switch (static_cast<::rund::host::Status>(value)) {
  case ::rund::host::Status::Ok:
  case ::rund::host::Status::Invalid:
  case ::rund::host::Status::CapacityExceeded:
  case ::rund::host::Status::SyscallFailed:
  case ::rund::host::Status::ReplayMismatch:
  case ::rund::host::Status::WouldBlock:
  case ::rund::host::Status::Unsupported:
    return true;
  }
  return false;
}

[[nodiscard]] std::uint16_t
  Fields(const ::rund::host::Event &event,
       const std::uint64_t logical_time_delta) noexcept {
  std::uint16_t fields = 0u;
  fields |= event_field(event.task_id != 0u, Task);
  fields |= event_field(logical_time_delta != 0u, LogicalTimeDelta);
  fields |= event_field(event.stream_id != 0u, Stream);
  fields |= event_field(event.draw_id != 0u, Draw);
  fields |= event_field(event.host_handle_id != 0u, Handle);
  fields |= event_field(event.offset != 0u, Offset);
  fields |= event_field(event.requested_bytes != 0u, Requested);
  fields |= event_field(event.completed_bytes != 0u, Completed);
  fields |= event_field(event.native_errno != 0, NativeErrno);
  fields |= event_field(event.name_hash.value != 0u, NameHash);
  fields |= event_field(event.path_hash.value != 0u, PathHash);
  fields |= event_field(event.payload_hash.value != 0u, PayloadHash);
  return fields;
}

[[nodiscard]] bool PresentValue(const std::uint16_t fields,
                                const EventField field,
                                const std::uint64_t value) noexcept {
  return (fields & field) == 0u ? value == 0u : value != 0u;
}

} // namespace

namespace replay_detail {

bool WriteHostEvidence(artifact::Writer &out,
                       const std::span<const ::rund::host::Event> events,
                       const std::uint64_t event_hash) noexcept {
  if (!out.ok()) {
    return false;
  }
  if (!out.varuint(static_cast<std::uint64_t>(events.size())) ||
      !out.fixed64(event_hash)) {
    return false;
  }
  std::uint64_t previous_sequence = 0u;
  std::uint64_t previous_logical_time = 0u;
  for (const ::rund::host::Event &event : events) {
    std::uint64_t sequence_delta = 0u;
    std::uint64_t logical_time_delta = 0u;
    if (!KnownKind(static_cast<std::uint64_t>(event.kind)) ||
        !KnownStatus(static_cast<std::uint64_t>(event.status)) ||
        !rund::kernel::checked::sub(event.sequence, previous_sequence,
                                    sequence_delta) ||
        !rund::kernel::checked::sub(
            event.logical_time_ns, previous_logical_time, logical_time_delta)) {
      return out.reject(::rund::replay::Code::CodecInvariantInvalid);
    }
    const std::uint16_t fields = Fields(event, logical_time_delta);
    if (!out.varuint(sequence_delta) ||
        !out.varuint(static_cast<std::uint64_t>(event.kind)) ||
        !out.varuint(static_cast<std::uint64_t>(event.status)) ||
        !out.varuint(fields) ||
        ((fields & Task) != 0u && !out.varuint(event.task_id)) ||
        ((fields & LogicalTimeDelta) != 0u &&
         !out.varuint(logical_time_delta)) ||
        ((fields & Stream) != 0u && !out.varuint(event.stream_id)) ||
        ((fields & Draw) != 0u && !out.varuint(event.draw_id)) ||
        ((fields & Handle) != 0u && !out.varuint(event.host_handle_id)) ||
        ((fields & Offset) != 0u && !out.varuint(event.offset)) ||
        ((fields & Requested) != 0u && !out.varuint(event.requested_bytes)) ||
        ((fields & Completed) != 0u && !out.varuint(event.completed_bytes)) ||
        ((fields & NativeErrno) != 0u && !out.sint(event.native_errno)) ||
        ((fields & NameHash) != 0u && !out.fixed64(event.name_hash.value)) ||
        ((fields & PathHash) != 0u && !out.fixed64(event.path_hash.value)) ||
        ((fields & PayloadHash) != 0u &&
         !out.fixed64(event.payload_hash.value))) {
      return false;
    }
    previous_sequence = event.sequence;
    previous_logical_time = event.logical_time_ns;
  }
  return true;
}

HostReplayDecodeResult ReadHostEvidence(artifact::Reader &in,
                                        const std::uint64_t max_entries) {
  HostReplayDecodeResult result{};
  std::uint64_t count = 0u;
  std::uint64_t expected_hash = 0u;
  if (!in.varuint(count)) {
    result.code = ::rund::replay::Code::HostBadValue;
    return result;
  }
  if (count > max_entries ||
      count >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    result.code = ::rund::replay::Code::CodecEntryCapacityExceeded;
    return result;
  }
  if (!in.fixed64(expected_hash)) {
    result.code = ::rund::replay::Code::HostBadValue;
    return result;
  }
  result.events.resize(static_cast<std::size_t>(count));
  std::uint64_t previous_sequence = 0u;
  std::uint64_t previous_logical_time = 0u;
  for (::rund::host::Event &event : result.events) {
    std::uint64_t sequence_delta = 0u;
    std::uint64_t kind = 0u;
    std::uint64_t status = 0u;
    std::uint64_t fields_value = 0u;
    if (!in.varuint(sequence_delta) || !in.varuint(kind) || !KnownKind(kind) ||
        !in.varuint(status) || !KnownStatus(status) ||
        !in.varuint(fields_value) || fields_value > kEventFieldMask) {
      result.code = ::rund::replay::Code::HostBadValue;
      return result;
    }
    const auto fields = static_cast<std::uint16_t>(fields_value);
    if (!rund::kernel::checked::add(previous_sequence, sequence_delta,
                                    event.sequence)) {
      result.code = ::rund::replay::Code::HostBadValue;
      return result;
    }
    event.kind = static_cast<::rund::host::EventKind>(kind);
    event.status = static_cast<::rund::host::Status>(status);
    std::uint64_t logical_time_delta = 0u;
    std::int64_t native_errno = 0;
    if (((fields & Task) != 0u && !in.varuint(event.task_id)) ||
        ((fields & LogicalTimeDelta) != 0u &&
         !in.varuint(logical_time_delta)) ||
        ((fields & Stream) != 0u && !in.varuint(event.stream_id)) ||
        ((fields & Draw) != 0u && !in.varuint(event.draw_id)) ||
        ((fields & Handle) != 0u && !in.varuint(event.host_handle_id)) ||
        ((fields & Offset) != 0u && !in.varuint(event.offset)) ||
        ((fields & Requested) != 0u && !in.varuint(event.requested_bytes)) ||
        ((fields & Completed) != 0u && !in.varuint(event.completed_bytes)) ||
        ((fields & NativeErrno) != 0u && !in.sint(native_errno)) ||
        native_errno < std::numeric_limits<std::int32_t>::min() ||
        native_errno > std::numeric_limits<std::int32_t>::max() ||
        ((fields & NameHash) != 0u && !in.fixed64(event.name_hash.value)) ||
        ((fields & PathHash) != 0u && !in.fixed64(event.path_hash.value)) ||
        ((fields & PayloadHash) != 0u &&
         !in.fixed64(event.payload_hash.value)) ||
        !PresentValue(fields, Task, event.task_id) ||
        !PresentValue(fields, LogicalTimeDelta, logical_time_delta) ||
        !PresentValue(fields, Stream, event.stream_id) ||
        !PresentValue(fields, Draw, event.draw_id) ||
        !PresentValue(fields, Handle, event.host_handle_id) ||
        !PresentValue(fields, Offset, event.offset) ||
        !PresentValue(fields, Requested, event.requested_bytes) ||
        !PresentValue(fields, Completed, event.completed_bytes) ||
        !PresentValue(fields, NativeErrno,
                      static_cast<std::uint64_t>(native_errno != 0)) ||
        !PresentValue(fields, NameHash, event.name_hash.value) ||
        !PresentValue(fields, PathHash, event.path_hash.value) ||
        !PresentValue(fields, PayloadHash, event.payload_hash.value)) {
      result.code = ::rund::replay::Code::HostBadValue;
      return result;
    }
    if (!rund::kernel::checked::add(previous_logical_time, logical_time_delta,
                                    event.logical_time_ns)) {
      result.code = ::rund::replay::Code::HostBadValue;
      return result;
    }
    event.native_errno = static_cast<std::int32_t>(native_errno);
    previous_sequence = event.sequence;
    previous_logical_time = event.logical_time_ns;
  }
  result.event_hash = ::rund::host::hash_events(result.events).value;
  if (result.event_hash != expected_hash) {
    result.code = ::rund::replay::Code::HostHashInvalid;
    return result;
  }
  result.code = ::rund::replay::Code::Ok;
  return result;
}

} // namespace replay_detail

std::vector<std::byte>
EncodeHostReplayEvents(const std::span<const ::rund::host::Event> events) {
  std::vector<std::byte> encoded{};
  try {
    encoded.reserve(replay_detail::artifact::kHeaderBytes +
                    sizeof(std::uint64_t) + events.size() * 8u);
    replay_detail::artifact::Writer out{replay_detail::artifact::Sink{
        .state = &encoded, .write = replay_detail::artifact::append}};
    if (!out.header(replay_detail::artifact::Kind::HostEvents) ||
        !replay_detail::WriteHostEvidence(
            out, events, ::rund::host::hash_events(events).value) ||
        !out.finish().ok()) {
      encoded.clear();
    }
  } catch (...) {
    encoded.clear();
  }
  return encoded;
}

replay_detail::HostReplayDecodeResult replay_detail::DecodeHostReplayEvents(
    const std::span<const std::byte> encoded) {
  try {
    artifact::Reader in{encoded};
    if (!in.header(artifact::Kind::HostEvents)) {
      return HostReplayDecodeResult{.code =
                                        ::rund::replay::Code::HostBadHeader};
    }
    HostReplayDecodeResult result =
        ReadHostEvidence(in, std::numeric_limits<std::uint64_t>::max());
    if (result.ok() && !in.done()) {
      result =
          HostReplayDecodeResult{.code = ::rund::replay::Code::HostBadValue};
    }
    return result;
  } catch (...) {
    return HostReplayDecodeResult{.code = ::rund::replay::Code::HostBadValue};
  }
}

bool DecodeHostReplayEvents(const std::span<const std::byte> encoded,
                            std::vector<::rund::host::Event> &out) {
  replay_detail::HostReplayDecodeResult decoded =
      replay_detail::DecodeHostReplayEvents(encoded);
  if (!decoded.ok()) {
    out.clear();
    return false;
  }
  out = std::move(decoded.events);
  return true;
}

} // namespace rund::node
