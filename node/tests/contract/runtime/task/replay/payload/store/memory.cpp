#include "test/assert.hpp"

#include "../../../coroutine/allocation.hpp"
#include "local/model.hpp"

#include <node/runtime/replay/host/payload.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace replay_payload_store {
namespace {

using rund::node::replay_detail::payload::Bytes;
using rund::node::replay_detail::payload::InputSourceRange;
using rund::node::replay_detail::payload::kChunkBytes;

static_assert(!std::is_copy_constructible_v<Store>);
static_assert(!std::is_copy_assignable_v<Store>);
static_assert(std::is_nothrow_move_constructible_v<Store>);
static_assert(std::is_nothrow_move_assignable_v<Store>);
static_assert(std::is_nothrow_copy_constructible_v<Bytes>);
static_assert(std::is_nothrow_copy_assignable_v<Bytes>);
static_assert(std::is_same_v<decltype(std::declval<const Bytes &>().span()),
                             std::span<const std::byte>>);
static_assert(std::is_same_v<decltype(std::declval<Bytes &>().data()),
                             const std::byte *>);

int BytesOwnValidRanges() {
  Bytes retained{};
  const std::byte *data = nullptr;
  {
    auto owner = std::make_shared<std::vector<std::byte>>(
        std::initializer_list<std::byte>{std::byte{0x10}, std::byte{0x20}});
    data = owner->data() + 1u;
    retained = Bytes::share(owner, 1u, 1u);
    TEST_ASSERT(retained.data() == data);
    TEST_ASSERT(retained.span()[0] == std::byte{0x20});
    TEST_ASSERT(retained.retained_bytes() == owner->capacity());

    const Bytes zero = Bytes::share(owner, owner->size(), 0u);
    TEST_ASSERT(zero.empty());
    TEST_ASSERT(zero.data() == nullptr);
    TEST_ASSERT(zero.retained_bytes() == 0u);
    const Bytes overflow = Bytes::share(owner, owner->size(), 1u);
    TEST_ASSERT(overflow.empty());
    TEST_ASSERT(overflow.data() == nullptr);
  }
  TEST_ASSERT(retained.data() == data);
  TEST_ASSERT(retained.span()[0] == std::byte{0x20});
  TEST_ASSERT(retained.slice(0u, 0u).data() == nullptr);
  TEST_ASSERT(retained.slice(1u, 1u).data() == nullptr);
  return 0;
}

int PreparedMultiChunkMetadataDoesNotAllocate() {
  constexpr std::size_t kBytes = kChunkBytes * 2u + 7u;
  Store store = Prepared(2u, 0u, kBytes);
  std::vector<std::byte> produced(kBytes);
  for (std::size_t index = 0u; index < produced.size(); ++index) {
    produced[index] =
        static_cast<std::byte>(((index * 131u) ^ (index >> 4u)) & 0xffu);
  }
  produced[kChunkBytes] ^= std::byte{0x01};
  produced[kChunkBytes * 2u] ^= std::byte{0x02};
  Bytes owner = Bytes::freeze(std::move(produced));
  const StableHash hash = hash_bytes(owner.data(), owner.size());
  const Capture capture = Capture::verify(owner.span(), hash);

  runtime_task_allocation::Start();
  const bool first = store.Append(1u, EventKind::IoRead, owner, capture);
  runtime_task_allocation::Stop();
  TEST_ASSERT(first);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(store.pieces(store.records().front()).size() == 3u);
  TEST_ASSERT(store.blobs().size() == 3u);

  store.Clear();
  runtime_task_allocation::Start();
  const bool second = store.Append(2u, EventKind::IoRead, owner, capture);
  runtime_task_allocation::Stop();
  TEST_ASSERT(second);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  return 0;
}

int StoreBoundsRecordsAndPiecesBeforeMutation() {
  constexpr std::uint64_t kBytes = kChunkBytes + 1u;
  const auto limits =
      rund::node::replay_detail::payload::Limits::runtime(1u, 1u, kBytes);
  TEST_ASSERT(limits.has_value());
  TEST_ASSERT(limits->pieces == 3u);
  TEST_ASSERT(!rund::node::replay_detail::payload::Limits::runtime(
                   std::numeric_limits<std::uint32_t>::max(), 1u, 0u)
                   .has_value());

  Store store = Prepared(1u, 1u, kBytes);
  const std::vector<std::byte> empty{};
  const StableHash empty_hash = hash_bytes(nullptr, 0u);
  TEST_ASSERT(
      store.Append(1u, EventKind::IoRead, Capture::verify(empty, empty_hash)));
  TEST_ASSERT(
      !store.Append(2u, EventKind::IoRead, Capture::verify(empty, empty_hash)));
  const auto source_hash = store.SourceRangeHash(0u, {}, 0u, 0u);
  TEST_ASSERT(source_hash.has_value());
  Bytes input = Bytes::freeze({});
  const InputSourceRange range{.hash = *source_hash};
  TEST_ASSERT(store.AppendInput(7u, 9u, 0u, range, input,
                                Capture::verify(input.span(), empty_hash)));
  TEST_ASSERT(!store.AppendInput(7u, 9u, 1u, range, input,
                                 Capture::verify(input.span(), empty_hash)));
  TEST_ASSERT(store.records().size() == 2u);
  TEST_ASSERT(store.blobs().empty());
  return 0;
}

} // namespace

int MemoryContract() {
  return Run(std::array<Contract, 3u>{
      BytesOwnValidRanges,
      PreparedMultiChunkMetadataDoesNotAllocate,
      StoreBoundsRecordsAndPiecesBeforeMutation,
  });
}

} // namespace replay_payload_store
