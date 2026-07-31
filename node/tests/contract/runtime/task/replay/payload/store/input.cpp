#include "test/assert.hpp"

#include "local/model.hpp"

#include <node/runtime/replay.hpp>
#include <node/runtime/replay/host/payload.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace replay_payload_store {
namespace {

using rund::node::replay_detail::BindPayloads;
using rund::node::replay_detail::payload::Bytes;
using rund::node::replay_detail::payload::ComputeSourceRangeHash;
using rund::node::replay_detail::payload::InputBinding;
using rund::node::replay_detail::payload::InputSourceRange;
using rund::node::replay_detail::payload::kChunkBytes;
using rund::node::replay_detail::payload::Materialize;
using rund::node::replay_detail::payload::Record;
using rund::node::replay_detail::payload::Role;
using rund::node::replay_detail::payload::SourcePayloadBinding;

int StoreAdoptsInputOwnerAndRejectsBadReplayIdentity() {
  std::vector<std::byte> produced(kChunkBytes + 7u);
  for (std::size_t index = 0u; index < produced.size(); ++index) {
    produced[index] = static_cast<std::byte>((index * 131u) & 0xffu);
  }
  const StableHash hash = hash_bytes(produced.data(), produced.size());
  const std::byte *const allocation = produced.data();
  Bytes owner = Bytes::freeze(std::move(produced));
  TEST_ASSERT(owner.data() == allocation);
  const Bytes returned = owner;

  Store store = Prepared();
  const auto source_hash = store.SourceRangeHash(0u, {}, 0u, 0u);
  TEST_ASSERT(source_hash.has_value());
  const InputSourceRange source_range{.hash = *source_hash};
  TEST_ASSERT(store.AppendInput(41u, 7u, 3u, source_range, owner,
                                Capture::verify(owner.span(), hash)));
  TEST_ASSERT(store.input_record_count() == 1u);
  TEST_ASSERT(store.host_record_count() == 0u);
  TEST_ASSERT(store.blobs().size() == 2u);
  TEST_ASSERT(store.blobs()[0].codec == Codec::Raw);
  TEST_ASSERT(store.blobs()[0].encoded.data() == allocation);
  TEST_ASSERT(store.blobs()[1].encoded.data() == allocation + kChunkBytes);
  TEST_ASSERT(returned.data() == allocation);
  const Record expected{
      .role = Role::Input,
      .input_source = 41u,
      .input_schema = 7u,
      .input_sequence = 3u,
      .source_hash = *source_hash,
      .completed_bytes = owner.size(),
      .payload_hash = hash,
  };
  TEST_ASSERT(store.records().front().metadata == expected);
  const Archive canonical = store.Archive();
  TEST_ASSERT(canonical.records.front().metadata == expected);
  TEST_ASSERT(Materialize(store).records.front().metadata == expected);

  const InputBinding binding{.source = 41u, .schema = 7u, .sequence = 3u};
  std::vector<std::byte> resolved(owner.size());
  TEST_ASSERT(store.ReadInput(0u, binding, resolved).ok());
  TEST_ASSERT(std::equal(resolved.begin(), resolved.end(), owner.span().begin(),
                         owner.span().end()));

  const auto wrong_order = store.ReadInput(
      0u, InputBinding{.source = 41u, .schema = 7u, .sequence = 4u}, resolved);
  TEST_ASSERT(!wrong_order.ok());
  TEST_ASSERT(wrong_order.code == rund::replay::Code::InputOrderMismatch);
  const auto wrong_schema = store.ReadInput(
      0u, InputBinding{.source = 41u, .schema = 8u, .sequence = 3u}, resolved);
  TEST_ASSERT(!wrong_schema.ok());
  TEST_ASSERT(wrong_schema.code == rund::replay::Code::InputOrderMismatch);

  Store changed_schema = Prepared();
  TEST_ASSERT(
      changed_schema.AppendInput(41u, 8u, 3u, source_range, returned,
                                 Capture::verify(returned.span(), hash)));
  TEST_ASSERT(changed_schema.payload_hash() != store.payload_hash());

  Archive corrupted = canonical;
  std::vector<std::byte> damaged(corrupted.chunks[0].encoded.span().begin(),
                                 corrupted.chunks[0].encoded.span().end());
  damaged[0] ^= std::byte{0x01};
  corrupted.chunks[0].encoded = Bytes::freeze(std::move(damaged));
  TEST_ASSERT(!Build(std::move(corrupted), {}).ok());
  return 0;
}

int SourceRangeValidationRejectsNonCanonicalBindings() {
  const std::vector<std::byte> first = Payload("first-source");
  const std::vector<std::byte> second = Payload("second-source");
  const std::vector<std::byte> input = Payload("input");
  const StableHash first_hash = hash_bytes(first.data(), first.size());
  const StableHash second_hash = hash_bytes(second.data(), second.size());
  const StableHash input_hash = hash_bytes(input.data(), input.size());
  const std::vector<Event> events{
      Event{.sequence = 1u,
            .kind = EventKind::IoRead,
            .status = Status::Ok,
            .requested_bytes = first.size(),
            .completed_bytes = first.size(),
            .payload_hash = first_hash},
      Event{.sequence = 2u, .kind = EventKind::EnvGet, .status = Status::Ok},
      Event{.sequence = 3u,
            .kind = EventKind::IoRead,
            .status = Status::Ok,
            .requested_bytes = second.size(),
            .completed_bytes = second.size(),
            .payload_hash = second_hash},
  };

  Store store = Prepared();
  TEST_ASSERT(
      store.Append(1u, EventKind::IoRead, Capture::verify(first, first_hash)));
  TEST_ASSERT(store.Append(3u, EventKind::IoRead,
                           Capture::verify(second, second_hash)));
  const auto source_hash = store.SourceRangeHash(0u, events, 0u, 2u);
  TEST_ASSERT(source_hash.has_value());
  Bytes input_owner = Bytes::freeze(std::vector<std::byte>{input});
  TEST_ASSERT(store.AppendInput(
      41u, 7u, 3u,
      InputSourceRange{.event_count = events.size(),
                       .payload_count = 2u,
                       .hash = *source_hash},
      input_owner, Capture::verify(input_owner.span(), input_hash)));

  const Archive valid = store.Archive();
  TEST_ASSERT(BindPayloads(events, valid) == rund::replay::Code::Ok);

  Archive out_of_order = valid;
  std::swap(out_of_order.records[0], out_of_order.records[1]);
  TEST_ASSERT(BindPayloads(events, out_of_order) ==
              rund::replay::Code::HostPayloadMissing);

  Archive missing = valid;
  missing.records.erase(missing.records.begin() + 1);
  TEST_ASSERT(BindPayloads(events, missing) ==
              rund::replay::Code::InputSourceHashMismatch);

  Archive duplicate = valid;
  const auto duplicate_host = duplicate.records.front();
  const auto input_record = duplicate.records.back();
  duplicate.records.pop_back();
  duplicate.records.insert(duplicate.records.begin() + 1, input_record);
  duplicate.records.insert(duplicate.records.begin() + 2, duplicate_host);
  TEST_ASSERT(BindPayloads(events, duplicate) ==
              rund::replay::Code::InputSourceHashMismatch);
  return 0;
}

int SourceRangeTraversalVisitsOnlySelectedRows() {
  constexpr std::size_t kEventCount = 4096u;
  constexpr std::size_t kOffset = 3072u;
  constexpr std::size_t kCount = 512u;
  std::vector<Event> events{};
  std::vector<SourcePayloadBinding> payloads{};
  events.reserve(kEventCount);
  payloads.reserve(kEventCount);
  for (std::size_t index = 0u; index < kEventCount; ++index) {
    const std::uint64_t sequence = static_cast<std::uint64_t>(index + 1u);
    const StableHash hash{.value = sequence * 17u};
    events.push_back(Event{.sequence = sequence,
                           .kind = EventKind::IoRead,
                           .status = Status::Ok,
                           .requested_bytes = 1u,
                           .completed_bytes = 1u,
                           .payload_hash = hash});
    payloads.push_back(SourcePayloadBinding{.event_sequence = sequence,
                                            .kind = EventKind::IoRead,
                                            .completed_bytes = 1u,
                                            .payload_hash = hash});
  }

  std::size_t cursor = kOffset;
  std::size_t calls = 0u;
  const auto next_payload =
      [&]() noexcept -> std::optional<SourcePayloadBinding> {
    ++calls;
    return cursor < payloads.size()
               ? std::optional<SourcePayloadBinding>{payloads[cursor++]}
               : std::nullopt;
  };
  const auto hash = ComputeSourceRangeHash(
      kOffset, std::span<const Event>{events}.subspan(kOffset, kCount), kOffset,
      kCount, next_payload);
  TEST_ASSERT(hash.has_value());
  TEST_ASSERT(calls == kCount);
  TEST_ASSERT(cursor == kOffset + kCount);
  return 0;
}

} // namespace

int InputContract() {
  return Run(std::array<Contract, 3u>{
      StoreAdoptsInputOwnerAndRejectsBadReplayIdentity,
      SourceRangeValidationRejectsNonCanonicalBindings,
      SourceRangeTraversalVisitsOnlySelectedRows,
  });
}

} // namespace replay_payload_store
