#include "test/assert.hpp"

#include "../local/model.hpp"

#include <rund/replay.hpp>

#include <algorithm>
#include <filesystem>
#include <span>
#include <vector>

namespace replay_spill {

int LifetimeContract() {
  const std::filesystem::path root = TempDir("record-lifetime");
  std::filesystem::remove_all(root);
  const std::vector<std::byte> first_bytes =
      Bytes("record one must survive record two");
  const std::vector<std::byte> second_bytes =
      Bytes("record two occupies another generation");

  rund::SessionConfig config{.workers = 1u};
  config.replay.storage = Storage(root, 1024u);
  config.replay.storage.max_allocated_bytes = 64u * 1024u;
  config.replay.storage.budget =
      ::rund::storage::Budget{config.replay.storage.max_allocated_bytes};
  config.scheduler.host_payload_capacity_bytes =
      config.replay.storage.max_bytes;
  config.replay.input_capacity = 1u;
  rund::Session session{};
  TEST_ASSERT(session.open(config));
  constexpr rund::replay::Input input{.id = 71u, .schema = 73u};
  std::size_t sources = 0u;
  const std::vector<std::byte> *next = &first_bytes;
  auto source = [&](rund::replay::Writer &writer) {
    ++sources;
    TEST_ASSERT(writer.append(*next));
    return static_cast<std::uint64_t>(sources);
  };
  const rund::replay::Binding binding{};
  const auto channel = binding.input(input, source);

  {
    const rund::replay::Record first =
        rund::replay::record(session, [&](rund::replay::Context &context) {
          const rund::replay::Value value = channel.read(context);
          TEST_ASSERT(value);
          TEST_ASSERT(std::equal(value.bytes().begin(), value.bytes().end(),
                                 first_bytes.begin(), first_bytes.end()));
        });
    TEST_ASSERT(first);
    next = &second_bytes;
    const rund::replay::Record second =
        rund::replay::record(session, [&](rund::replay::Context &context) {
          const rund::replay::Value value = channel.read(context);
          TEST_ASSERT(value);
          TEST_ASSERT(std::equal(value.bytes().begin(), value.bytes().end(),
                                 second_bytes.begin(), second_bytes.end()));
        });
    TEST_ASSERT(second);
    TEST_ASSERT(sources == 2u);
    TEST_ASSERT(GenerationDirectories(root).size() == 2u);

    std::vector<std::byte> artifact{};
    TEST_ASSERT(
        first.save([&](const std::span<const std::byte> bytes) noexcept {
          try {
            artifact.insert(artifact.end(), bytes.begin(), bytes.end());
            return true;
          } catch (...) {
            return false;
          }
        }));
    TEST_ASSERT(!artifact.empty());

    bool observed = false;
    const rund::replay::Check replayed =
        rund::replay::run(session, first, [&](rund::replay::Context &context) {
          const rund::replay::Value value = channel.read(context);
          observed =
              value && std::equal(value.bytes().begin(), value.bytes().end(),
                                  first_bytes.begin(), first_bytes.end());
        });
    TEST_ASSERT(replayed);
    TEST_ASSERT(observed);
    TEST_ASSERT(sources == 2u);
  }
  TEST_ASSERT(GenerationDirectories(root).empty());
  TEST_ASSERT(session.close());
  std::filesystem::remove_all(root);
  return 0;
}

} // namespace replay_spill
