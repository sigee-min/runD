#include "../store.hpp"

#include <utility>

namespace rund::node::replay_detail::payload {

bool Store::Append(const std::uint64_t event_sequence,
                   const ::rund::host::EventKind kind, const Capture payload) {
  if (!payload) {
    return false;
  }
  const std::span<const std::byte> bytes = payload.bytes();
  return AppendRecord(
      StoredRecord{
          .metadata =
              {
                  .role = ::rund::node::replay_detail::payload::Role::Host,
                  .event_sequence = event_sequence,
                  .kind = kind,
                  .completed_bytes = bytes.size(),
                  .payload_hash = payload.hash(),
              },
      },
      payload);
}

bool Store::Append(const std::uint64_t event_sequence,
                   const ::rund::host::EventKind kind,
                   ::rund::node::replay_detail::payload::Bytes bytes,
                   const Capture payload) {
  if (!payload) {
    return false;
  }
  const std::span<const std::byte> span = bytes.span();
  return AppendRecord(
      StoredRecord{
          .metadata =
              {
                  .role = ::rund::node::replay_detail::payload::Role::Host,
                  .event_sequence = event_sequence,
                  .kind = kind,
                  .completed_bytes = span.size(),
                  .payload_hash = payload.hash(),
              },
      },
      payload, std::move(bytes));
}

bool Store::AppendInput(const std::uint64_t source, const std::uint64_t schema,
                        const std::uint64_t sequence,
                        const InputSourceRange source_range,
                        ::rund::node::replay_detail::payload::Bytes bytes,
                        const Capture payload) {
  if (source == 0u || schema == 0u || !payload) {
    return false;
  }
  const std::span<const std::byte> span = bytes.span();
  return AppendRecord(
      StoredRecord{
          .metadata =
              {
                  .role = ::rund::node::replay_detail::payload::Role::Input,
                  .input_source = source,
                  .input_schema = schema,
                  .input_sequence = sequence,
                  .source_event_offset = source_range.event_offset,
                  .source_event_count = source_range.event_count,
                  .source_payload_offset = source_range.payload_offset,
                  .source_payload_count = source_range.payload_count,
                  .source_hash = source_range.hash,
                  .completed_bytes = span.size(),
                  .payload_hash = payload.hash(),
              },
      },
      payload, std::move(bytes));
}

bool Store::CapturesIngress() const noexcept { return diagnostic_.enabled(); }

::rund::StableHash Store::CaptureIngress(const std::uint64_t event_sequence,
                                         const ::rund::host::EventKind kind,
                                         const RawByteSource &source) noexcept {
  loaded_diagnostic_active_ = false;
  loaded_diagnostic_ = {};
  return diagnostic_.Capture(event_sequence, kind, source);
}

} // namespace rund::node::replay_detail::payload
