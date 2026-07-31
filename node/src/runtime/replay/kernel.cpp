#include <node/runtime/replay/kernel.hpp>

#include "../../host/hash/fields.hpp"
#include "artifact/bytes.hpp"
#include "artifact/format.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace rund::node::replay {
namespace {

[[nodiscard]] bool RequiredIdentityPresent(const KernelDesc &desc) noexcept {
  return desc.run_key_hash != 0u && desc.program_hash != 0u &&
         desc.phase_hash != 0u && desc.dispatch_hash != 0u &&
         desc.reduction_hash != 0u && desc.capacity_hash != 0u &&
         desc.output_hash != 0u;
}

[[nodiscard]] bool
RequiredIdentityPresent(const KernelRecord &record) noexcept {
  return record.run_key_hash != 0u && record.program_hash != 0u &&
         record.phase_hash != 0u && record.dispatch_hash != 0u &&
         record.reduction_hash != 0u && record.capacity_hash != 0u &&
         record.output_hash != 0u;
}

[[nodiscard]] ::rund::replay::Code
ValidateKernelRecordFields(const KernelRecord &record) noexcept {
  if (!RequiredIdentityPresent(record)) {
    return ::rund::replay::Code::KernelIdentityMissing;
  }
  if (!::rund::replay::valid(record.code) ||
      (record.code != ::rund::replay::Code::Ok &&
       ::rund::replay::family(record.code) != ::rund::replay::Family::Kernel)) {
    return ::rund::replay::Code::KernelBadValue;
  }
  if (record.record_hash != HashKernelRecord(record)) {
    return ::rund::replay::Code::KernelHashInvalid;
  }
  return ::rund::replay::Code::Ok;
}

void Reject(KernelRecord &out, const ::rund::replay::Code code) {
  out = KernelRecord{};
  out.code = code;
}

} // namespace

KernelRecord MakeKernelRecord(const KernelDesc desc) {
  KernelRecord record{};
  if (!RequiredIdentityPresent(desc)) {
    record.code = ::rund::replay::Code::KernelIdentityMissing;
    return record;
  }
  if (!::rund::replay::valid(desc.code) ||
      (desc.code != ::rund::replay::Code::Ok &&
       ::rund::replay::family(desc.code) != ::rund::replay::Family::Kernel)) {
    record.code = ::rund::replay::Code::KernelBadValue;
    return record;
  }

  record.code = desc.code;
  record.run_key_hash = desc.run_key_hash;
  record.program_hash = desc.program_hash;
  record.phase_hash = desc.phase_hash;
  record.dispatch_hash = desc.dispatch_hash;
  record.reduction_hash = desc.reduction_hash;
  record.capacity_hash = desc.capacity_hash;
  record.output_hash = desc.output_hash;
  record.record_hash = HashKernelRecord(record);
  return record;
}

bool IsValidKernelRecord(const KernelRecord &record) noexcept {
  return ValidateKernelRecordFields(record) == ::rund::replay::Code::Ok;
}

std::uint64_t HashKernelRecord(const KernelRecord &record) noexcept {
  constexpr std::uint64_t kScalarBytes = 8u * sizeof(std::uint64_t);
  host_detail::CanonicalByteHasher hash(kScalarBytes);
  hash.AppendU64Le(::rund::replay::raw(record.code));
  hash.AppendU64Le(record.run_key_hash);
  hash.AppendU64Le(record.program_hash);
  hash.AppendU64Le(record.phase_hash);
  hash.AppendU64Le(record.dispatch_hash);
  hash.AppendU64Le(record.reduction_hash);
  hash.AppendU64Le(record.capacity_hash);
  hash.AppendU64Le(record.output_hash);
  return hash.Finish().value;
}

std::uint64_t
HashKernelRecords(const std::span<const KernelRecord> records) noexcept {
  host_detail::CanonicalByteHasher hash(
      (static_cast<std::uint64_t>(records.size()) + 1u) *
      sizeof(std::uint64_t));
  hash.AppendU64Le(static_cast<std::uint64_t>(records.size()));
  for (const KernelRecord &record : records) {
    hash.AppendU64Le(HashKernelRecord(record));
  }
  return hash.Finish().value;
}

std::vector<std::byte> EncodeKernelRecord(const KernelRecord &record) {
  std::vector<std::byte> encoded{};
  encoded.reserve(80u);
  replay_detail::artifact::Writer out{replay_detail::artifact::Sink{
      .state = &encoded, .write = replay_detail::artifact::append}};
  if (!out.header(replay_detail::artifact::Kind::Kernel) ||
      !out.varuint(::rund::replay::raw(record.code)) ||
      !out.fixed64(record.run_key_hash) || !out.fixed64(record.program_hash) ||
      !out.fixed64(record.phase_hash) || !out.fixed64(record.dispatch_hash) ||
      !out.fixed64(record.reduction_hash) ||
      !out.fixed64(record.capacity_hash) || !out.fixed64(record.output_hash) ||
      !out.fixed64(record.record_hash) || !out.finish().ok()) {
    encoded.clear();
  }
  return encoded;
}

bool DecodeKernelRecord(const std::span<const std::byte> encoded,
                        KernelRecord &out) {
  replay_detail::artifact::Reader in{encoded};
  if (!in.header(replay_detail::artifact::Kind::Kernel)) {
    Reject(out, ::rund::replay::Code::KernelBadHeader);
    return false;
  }
  std::uint64_t code = 0u;
  KernelDesc desc{};
  std::uint64_t record_hash = 0u;
  if (!in.varuint(code) || code > std::numeric_limits<std::uint32_t>::max() ||
      !in.fixed64(desc.run_key_hash) || !in.fixed64(desc.program_hash) ||
      !in.fixed64(desc.phase_hash) || !in.fixed64(desc.dispatch_hash) ||
      !in.fixed64(desc.reduction_hash) || !in.fixed64(desc.capacity_hash) ||
      !in.fixed64(desc.output_hash) || !in.fixed64(record_hash) || !in.done()) {
    Reject(out, ::rund::replay::Code::KernelBadValue);
    return false;
  }
  desc.code =
      static_cast<::rund::replay::Code>(static_cast<std::uint32_t>(code));
  KernelRecord record = MakeKernelRecord(desc);
  if (!IsValidKernelRecord(record) || record.record_hash != record_hash) {
    Reject(out, IsValidKernelRecord(record)
                    ? ::rund::replay::Code::KernelHashInvalid
                    : record.code);
    return false;
  }
  out = record;
  return true;
}

KernelCheck CheckKernelRecord(const KernelRecord &expected,
                              const KernelRecord &actual) noexcept {
  const std::uint64_t expected_hash = HashKernelRecord(expected);
  const std::uint64_t actual_hash = HashKernelRecord(actual);
  if (expected_hash != actual_hash) {
    return KernelCheck{.code = ::rund::replay::Code::KernelMismatch,
                       .expected_hash = expected_hash,
                       .actual_hash = actual_hash};
  }
  if (expected.record_hash != actual.record_hash ||
      expected.record_hash != expected_hash ||
      actual.record_hash != actual_hash) {
    return KernelCheck{.code = ::rund::replay::Code::KernelHashMismatch,
                       .expected_hash = expected.record_hash,
                       .actual_hash = actual.record_hash};
  }
  return KernelCheck{.code = ::rund::replay::Code::Ok,
                     .expected_hash = expected_hash,
                     .actual_hash = actual_hash};
}

} // namespace rund::node::replay
