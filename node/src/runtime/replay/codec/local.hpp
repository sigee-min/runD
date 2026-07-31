#pragma once

#include "../artifact/format.hpp"
#include <kernel/core/checked.hpp>

#include <rund/replay/limits.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace rund {
struct Trace;
namespace host {
struct Event;
}
namespace task {
struct Observation;
}
} // namespace rund

namespace rund::node::replay_detail::payload {
struct Archive;
class Store;
} // namespace rund::node::replay_detail::payload

namespace rund::node::replay_detail::artifact {

class Admission final {
public:
  explicit Admission(const ::rund::replay::Limits limits) noexcept
      : limits_(limits) {}

  [[nodiscard]] bool entries(const std::uint64_t count) noexcept {
    std::uint64_t next = 0u;
    if (!rund::kernel::checked::add(entries_, count, next) || next > limits_.max_entries) {
      code_ = ::rund::replay::Code::CodecEntryCapacityExceeded;
      return false;
    }
    entries_ = next;
    return true;
  }

  [[nodiscard]] bool payload(std::uint64_t &total,
                             const std::uint64_t count) noexcept {
    std::uint64_t next = 0u;
    if (!rund::kernel::checked::add(total, count, next) || next > limits_.max_payload_bytes) {
      code_ = ::rund::replay::Code::CodecPayloadCapacityExceeded;
      return false;
    }
    total = next;
    return true;
  }

  [[nodiscard]] const ::rund::replay::Limits &limits() const noexcept {
    return limits_;
  }

  [[nodiscard]] ::rund::replay::Code
  failure(const ::rund::replay::Code malformed) const noexcept {
    return code_ == ::rund::replay::Code::Ok ? malformed : code_;
  }

private:
  ::rund::replay::Limits limits_{};
  std::uint64_t entries_ = 0u;
  ::rund::replay::Code code_ = ::rund::replay::Code::Ok;
};

enum RecordField : std::uint16_t {
  InputRole = 1u << 0u,
  SourceChanged = 1u << 1u,
  SchemaChanged = 1u << 2u,
  SequenceExceptional = 1u << 3u,
  EventRangePresent = 1u << 4u,
  PayloadRangePresent = 1u << 5u,
  SourceHashStored = 1u << 6u,
  PieceCountExceptional = 1u << 7u,
  ChunkChanged = 1u << 8u,
  PayloadHashStored = 1u << 9u,
};

inline constexpr std::uint16_t kRecordFieldMask =
    InputRole | SourceChanged | SchemaChanged | SequenceExceptional |
    EventRangePresent | PayloadRangePresent | SourceHashStored |
    PieceCountExceptional | ChunkChanged | PayloadHashStored;

[[nodiscard]] inline bool size(const std::uint64_t value,
                               std::size_t &out) noexcept {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  out = static_cast<std::size_t>(value);
  return true;
}

[[nodiscard]] inline bool runtime_code(const std::uint64_t raw,
                                       ::rund::replay::Code &code) noexcept {
  if (raw > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  code = static_cast<::rund::replay::Code>(static_cast<std::uint32_t>(raw));
  return ::rund::replay::valid(code) &&
         (code == ::rund::replay::Code::Ok ||
          ::rund::replay::family(code) == ::rund::replay::Family::Runtime);
}

[[nodiscard]] bool
write_observations(Writer &out,
                   std::span<const task::Observation> observations) noexcept;
[[nodiscard]] bool read_observations(Reader &in, Admission &admission,
                                     std::vector<task::Observation> &out);
[[nodiscard]] bool write_trace(Writer &out,
                               const ::rund::Trace &trace) noexcept;
[[nodiscard]] bool read_trace(Reader &in, Admission &admission,
                              ::rund::Trace &trace);

[[nodiscard]] bool write_payload(Writer &out, const payload::Archive &archive,
                                 const payload::Store *payloads,
                                 bool required_by_host_events);
[[nodiscard]] bool read_payload(Reader &in, Admission &admission,
                                std::span<const ::rund::host::Event> events,
                                payload::Archive &archive);

} // namespace rund::node::replay_detail::artifact
