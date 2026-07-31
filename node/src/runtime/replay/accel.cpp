#include <node/runtime/replay/accel.hpp>

#include "../../host/hash/fields.hpp"
#include "artifact/bytes.hpp"
#include "artifact/format.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::replay {
namespace {

[[nodiscard]] bool RequiredIdentityPresent(const AccelDesc &desc) noexcept {
  return desc.graph_hash != 0u && desc.kernel_hash != 0u &&
         desc.backend_hash != 0u && desc.caps_hash != 0u &&
         desc.binding_hash != 0u && desc.buffer_shape_hash != 0u &&
         desc.dispatch_hash != 0u && desc.output_hash != 0u;
}

[[nodiscard]] bool RequiredIdentityPresent(const AccelRecord &record) noexcept {
  return record.graph_hash != 0u && record.kernel_hash != 0u &&
         record.backend_hash != 0u && record.caps_hash != 0u &&
         record.binding_hash != 0u && record.buffer_shape_hash != 0u &&
         record.dispatch_hash != 0u && record.output_hash != 0u;
}

[[nodiscard]] std::uint64_t
HashAccelSemantic(const AccelRecord &record) noexcept {
  constexpr std::uint64_t kScalarBytes = 9u * sizeof(std::uint64_t);
  host_detail::CanonicalByteHasher hash(kScalarBytes);
  hash.AppendU64Le(::rund::replay::raw(record.code));
  hash.AppendU64Le(record.graph_hash);
  hash.AppendU64Le(record.kernel_hash);
  hash.AppendU64Le(record.backend_hash);
  hash.AppendU64Le(record.caps_hash);
  hash.AppendU64Le(record.binding_hash);
  hash.AppendU64Le(record.buffer_shape_hash);
  hash.AppendU64Le(record.dispatch_hash);
  hash.AppendU64Le(record.output_hash);
  return hash.Finish().value;
}

[[nodiscard]] std::uint64_t
HashAccelDiagnostic(const AccelRecord &record) noexcept {
  host_detail::CanonicalByteHasher hash(5u * sizeof(std::uint64_t));
  hash.AppendU64Le(record.semantic_hash);
  hash.AppendU64Le(record.diagnostic_runtime_ns);
  hash.AppendU64Le(record.diagnostic_driver_hash);
  hash.AppendU64Le(record.diagnostic_cache_hash);
  hash.AppendU64Le(record.diagnostic_submit_count);
  return hash.Finish().value;
}

[[nodiscard]] ::rund::replay::Code
ValidateAccelRecordFields(const AccelRecord &record) noexcept {
  if (!RequiredIdentityPresent(record)) {
    return ::rund::replay::Code::AccelIdentityMissing;
  }
  if (!::rund::replay::valid(record.code) ||
      (record.code != ::rund::replay::Code::Ok &&
       ::rund::replay::family(record.code) != ::rund::replay::Family::Accel)) {
    return ::rund::replay::Code::AccelBadValue;
  }
  if (record.semantic_hash != HashAccelSemantic(record) ||
      record.record_hash != record.semantic_hash ||
      record.diagnostic_hash != HashAccelDiagnostic(record)) {
    return ::rund::replay::Code::AccelHashInvalid;
  }
  return ::rund::replay::Code::Ok;
}

void Reject(AccelRecord &out, const ::rund::replay::Code code) {
  out = AccelRecord{};
  out.code = code;
}

} // namespace

AccelRecord MakeAccelRecord(const AccelDesc desc) {
  AccelRecord record{};
  if (!RequiredIdentityPresent(desc)) {
    record.code = ::rund::replay::Code::AccelIdentityMissing;
    return record;
  }
  if (!::rund::replay::valid(desc.code) ||
      (desc.code != ::rund::replay::Code::Ok &&
       ::rund::replay::family(desc.code) != ::rund::replay::Family::Accel)) {
    record.code = ::rund::replay::Code::AccelBadValue;
    return record;
  }

  record.code = desc.code;
  record.graph_hash = desc.graph_hash;
  record.kernel_hash = desc.kernel_hash;
  record.backend_hash = desc.backend_hash;
  record.caps_hash = desc.caps_hash;
  record.binding_hash = desc.binding_hash;
  record.buffer_shape_hash = desc.buffer_shape_hash;
  record.dispatch_hash = desc.dispatch_hash;
  record.output_hash = desc.output_hash;
  record.diagnostic_runtime_ns = desc.diagnostic_runtime_ns;
  record.diagnostic_driver_hash = desc.diagnostic_driver_hash;
  record.diagnostic_cache_hash = desc.diagnostic_cache_hash;
  record.diagnostic_submit_count = desc.diagnostic_submit_count;
  record.semantic_hash = HashAccelSemantic(record);
  record.diagnostic_hash = HashAccelDiagnostic(record);
  record.record_hash = record.semantic_hash;
  return record;
}

bool IsValidAccelRecord(const AccelRecord &record) noexcept {
  return ValidateAccelRecordFields(record) == ::rund::replay::Code::Ok;
}

std::uint64_t HashAccelRecord(const AccelRecord &record) noexcept {
  return HashAccelSemantic(record);
}

std::uint64_t
HashAccelRecords(const std::span<const AccelRecord> records) noexcept {
  host_detail::CanonicalByteHasher hash(
      (static_cast<std::uint64_t>(records.size()) + 1u) *
      sizeof(std::uint64_t));
  hash.AppendU64Le(static_cast<std::uint64_t>(records.size()));
  for (const AccelRecord &record : records) {
    hash.AppendU64Le(HashAccelRecord(record));
  }
  return hash.Finish().value;
}

std::vector<std::byte> EncodeAccelRecord(const AccelRecord &record) {
  std::vector<std::byte> encoded{};
  encoded.reserve(128u);
  replay_detail::artifact::Writer out{replay_detail::artifact::Sink{
      .state = &encoded, .write = replay_detail::artifact::append}};
  if (!out.header(replay_detail::artifact::Kind::Accelerator) ||
      !out.varuint(::rund::replay::raw(record.code)) ||
      !out.fixed64(record.graph_hash) || !out.fixed64(record.kernel_hash) ||
      !out.fixed64(record.backend_hash) || !out.fixed64(record.caps_hash) ||
      !out.fixed64(record.binding_hash) ||
      !out.fixed64(record.buffer_shape_hash) ||
      !out.fixed64(record.dispatch_hash) || !out.fixed64(record.output_hash) ||
      !out.varuint(record.diagnostic_runtime_ns) ||
      !out.fixed64(record.diagnostic_driver_hash) ||
      !out.fixed64(record.diagnostic_cache_hash) ||
      !out.varuint(record.diagnostic_submit_count) ||
      !out.fixed64(record.semantic_hash) ||
      !out.fixed64(record.diagnostic_hash) || !out.finish().ok()) {
    encoded.clear();
  }
  return encoded;
}

bool DecodeAccelRecord(const std::span<const std::byte> encoded,
                       AccelRecord &out) {
  replay_detail::artifact::Reader in{encoded};
  if (!in.header(replay_detail::artifact::Kind::Accelerator)) {
    Reject(out, ::rund::replay::Code::AccelBadHeader);
    return false;
  }
  std::uint64_t code = 0u;
  AccelDesc desc{};
  std::uint64_t semantic_hash = 0u;
  std::uint64_t diagnostic_hash = 0u;
  if (!in.varuint(code) || code > std::numeric_limits<std::uint32_t>::max() ||
      !in.fixed64(desc.graph_hash) || !in.fixed64(desc.kernel_hash) ||
      !in.fixed64(desc.backend_hash) || !in.fixed64(desc.caps_hash) ||
      !in.fixed64(desc.binding_hash) || !in.fixed64(desc.buffer_shape_hash) ||
      !in.fixed64(desc.dispatch_hash) || !in.fixed64(desc.output_hash) ||
      !in.varuint(desc.diagnostic_runtime_ns) ||
      !in.fixed64(desc.diagnostic_driver_hash) ||
      !in.fixed64(desc.diagnostic_cache_hash) ||
      !in.varuint(desc.diagnostic_submit_count) || !in.fixed64(semantic_hash) ||
      !in.fixed64(diagnostic_hash) || !in.done()) {
    Reject(out, ::rund::replay::Code::AccelBadValue);
    return false;
  }
  desc.code =
      static_cast<::rund::replay::Code>(static_cast<std::uint32_t>(code));
  AccelRecord record = MakeAccelRecord(desc);
  if (!IsValidAccelRecord(record) || record.semantic_hash != semantic_hash ||
      record.diagnostic_hash != diagnostic_hash) {
    Reject(out, IsValidAccelRecord(record)
                    ? ::rund::replay::Code::AccelHashInvalid
                    : record.code);
    return false;
  }
  out = record;
  return true;
}

AccelCheck CheckAccelRecord(const AccelRecord &expected,
                            const AccelRecord &actual) noexcept {
  const std::uint64_t expected_hash = HashAccelSemantic(expected);
  const std::uint64_t actual_hash = HashAccelSemantic(actual);
  if (expected_hash != actual_hash) {
    return AccelCheck{.code = ::rund::replay::Code::AccelMismatch,
                      .expected_hash = expected_hash,
                      .actual_hash = actual_hash};
  }
  if (expected.semantic_hash != actual.semantic_hash ||
      expected.record_hash != actual.record_hash ||
      expected.semantic_hash != expected_hash ||
      actual.semantic_hash != actual_hash) {
    return AccelCheck{.code = ::rund::replay::Code::AccelHashMismatch,
                      .expected_hash = expected.semantic_hash,
                      .actual_hash = actual.semantic_hash};
  }
  return AccelCheck{.code = ::rund::replay::Code::Ok,
                    .expected_hash = expected_hash,
                    .actual_hash = actual_hash};
}

} // namespace rund::node::replay
