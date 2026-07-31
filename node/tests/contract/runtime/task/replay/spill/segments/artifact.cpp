#include "test/assert.hpp"

#include "../local/model.hpp"

#include <rund/replay.hpp>

#include <algorithm>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace replay_spill {

int ArtifactContract() {
  const std::filesystem::path dir = TempDir("artifact");
  std::filesystem::remove_all(dir);
  std::vector<std::byte> payload(257u);
  for (std::size_t index = 0u; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>((index * 73u + 19u) & 0xffu);
  }
  rund::SessionConfig config{.workers = 1u};
  config.replay.storage = Storage(dir, 1024u);
  config.scheduler.host_payload_capacity_bytes =
      config.replay.storage.max_bytes;
  config.replay.input_capacity = 4u;
  rund::Session session{};
  TEST_ASSERT(session.open(config));
  constexpr rund::replay::Input input{.id = 17u, .schema = 23u};
  std::size_t producers = 0u;
  bool observed = false;
  auto restore = [](std::span<const std::byte>) {
    return rund::replay::Restore::Restored;
  };
  rund::replay::Binding replay_binding{0x7301u, restore};
  auto source = [&](rund::replay::Writer &writer) {
    ++producers;
    TEST_ASSERT(writer.append(payload));
    return std::uint64_t{5u};
  };
  const auto channel = replay_binding.input(input, source);
  const rund::replay::Record record =
      rund::replay::record(session, [&](rund::replay::Context &context) {
        const rund::replay::Value value = channel.read(context);
        observed =
            value && std::equal(value.bytes().begin(), value.bytes().end(),
                                payload.begin(), payload.end());
      });
  TEST_ASSERT(record);
  TEST_ASSERT(observed);
  TEST_ASSERT(producers == 1u);
  TEST_ASSERT(record.storage_report().mode == rund::replay::StorageMode::Spill);

  std::vector<std::byte> artifact{};
  std::size_t largest_write = 0u;
  const rund::replay::Save saved =
      record.save([&](const std::span<const std::byte> bytes) noexcept {
        try {
          largest_write = std::max(largest_write, bytes.size());
          artifact.insert(artifact.end(), bytes.begin(), bytes.end());
          return true;
        } catch (...) {
          return false;
        }
      });
  TEST_ASSERT(saved);
  TEST_ASSERT(saved.bytes() == artifact.size());
  TEST_ASSERT(saved.writes() != 0u);
  TEST_ASSERT(largest_write <= 4096u);
  const auto rejecting_sink = [](std::span<const std::byte>) noexcept {
    return false;
  };
  const rund::replay::Save rejected = record.save(rejecting_sink);
  TEST_ASSERT(!rejected);
  TEST_ASSERT(rejected.code() == rund::replay::Code::ArtifactWriteFailed);
  auto throwing_sink = [](std::span<const std::byte>) -> bool {
    throw std::runtime_error{"sink failed"};
  };
  static_assert(noexcept(record.save(throwing_sink)));
  static_assert(
      noexcept(rund::replay::Record::load(std::span<const std::byte>{})));
  const rund::replay::Save threw = record.save(throwing_sink);
  TEST_ASSERT(!threw);
  TEST_ASSERT(threw.code() == rund::replay::Code::ArtifactWriteFailed);

  TEST_ASSERT(session.close());
  std::filesystem::remove_all(dir);
  const rund::replay::Load<rund::replay::Record> loaded =
      rund::replay::Record::load(artifact);
  TEST_ASSERT(loaded);
  TEST_ASSERT(loaded->storage_report().mode ==
              rund::replay::StorageMode::Memory);
  TEST_ASSERT(loaded->storage_report().retained_bytes ==
              loaded->storage_report().encoded_bytes);
  TEST_ASSERT(rund::replay::check(record, *loaded));

  rund::Session replay{};
  rund::SessionConfig memory{.workers = 1u};
  memory.replay.storage.max_bytes = config.replay.storage.max_bytes;
  memory.scheduler.host_payload_capacity_bytes =
      config.replay.storage.max_bytes;
  memory.replay.input_capacity = 4u;
  TEST_ASSERT(replay.open(memory));
  producers = 0u;
  observed = false;
  const rund::replay::Check checked =
      rund::replay::run(replay, *loaded, [&](rund::replay::Context &context) {
        const rund::replay::Value value = channel.read(context);
        observed =
            value && std::equal(value.bytes().begin(), value.bytes().end(),
                                payload.begin(), payload.end());
      });
  TEST_ASSERT(checked);
  TEST_ASSERT(observed);
  TEST_ASSERT(producers == 0u);
  TEST_ASSERT(replay.close());
  return 0;
}

} // namespace replay_spill
