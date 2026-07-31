#include "local/model.hpp"

#include "test/assert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace runtime_task_replay_run {

int Capacity(rund::SessionConfig config) {
  config.scheduler.host_payload_capacity_bytes = 1024u;
  config.replay.storage.max_bytes = 1024u;

  {
    config.replay.input_capacity = 0u;
    rund::Session session;
    TEST_ASSERT(session.open(config));
    std::size_t producers = 0u;
    auto restore = [](std::span<const std::byte>) {
      return rund::replay::Restore::Restored;
    };
    rund::replay::Binding binding{kStateSchema, restore};
    auto source = [&](rund::replay::Writer &writer) {
      ++producers;
      Append(writer, std::array{std::byte{0x01}});
      return std::uint64_t{0u};
    };
    const auto channel = binding.input(kInput, source);
    rund::replay::Code value_code = rund::replay::Code::Ok;
    const rund::replay::Record disabled =
        rund::replay::record(session, [&](rund::replay::Context &input) {
          const rund::replay::Value value = channel.read(input);
          value_code = value.code();
        });
    TEST_ASSERT(!disabled);
    TEST_ASSERT(producers == 0u);
    TEST_ASSERT(value_code == rund::replay::Code::InputCapacityExceeded);
    TEST_ASSERT(session.close());
  }

  {
    config.replay.input_capacity = 1u;
    rund::Session session;
    TEST_ASSERT(session.open(config));
    std::size_t producers = 0u;
    std::uint64_t sequence = 0u;
    auto restore = [](std::span<const std::byte>) {
      return rund::replay::Restore::Restored;
    };
    rund::replay::Binding binding{kStateSchema, restore};
    auto source = [&](rund::replay::Writer &writer) {
      ++producers;
      Append(writer, std::array{std::byte{0x02}});
      return sequence++;
    };
    const auto channel = binding.input(kInput, source);
    rund::replay::Code overflow_code = rund::replay::Code::Ok;
    const rund::replay::Record overflow =
        rund::replay::record(session, [&](rund::replay::Context &input) {
          const rund::replay::Value first = channel.read(input);
          TEST_ASSERT(first);
          TEST_ASSERT(first.sequence() == 0u);
          const rund::replay::Value second = channel.read(input);
          overflow_code = second.code();
        });
    TEST_ASSERT(!overflow);
    TEST_ASSERT(producers == 1u);
    TEST_ASSERT(overflow_code == rund::replay::Code::InputCapacityExceeded);
    TEST_ASSERT(session.close());
  }

  auto record_with = [&](const std::uint32_t capacity) {
    config.replay.input_capacity = capacity;
    rund::Session session;
    TEST_ASSERT(session.open(config));
    std::uint64_t sequence = 0u;
    auto restore = [](std::span<const std::byte>) {
      return rund::replay::Restore::Restored;
    };
    rund::replay::Binding binding{kStateSchema, restore};
    auto source = [&](rund::replay::Writer &writer) {
      Append(writer, std::array{std::byte{0x03}});
      return sequence++;
    };
    const auto channel = binding.input(kInput, source);
    const rund::replay::Record captured =
        rund::replay::record(session, [&](rund::replay::Context &input) {
          const rund::replay::Value first = channel.read(input);
          const rund::replay::Value second = channel.read(input);
          TEST_ASSERT(first && first.sequence() == 0u);
          TEST_ASSERT(second && second.sequence() == 1u);
        });
    TEST_ASSERT(captured);
    TEST_ASSERT(session.close());
    return captured;
  };

  const rund::replay::Record two = record_with(2u);
  const rund::replay::Record three = record_with(3u);
  TEST_ASSERT(two.hash() == three.hash());

  {
    config.replay.input_capacity = 2u;
    rund::Session session;
    TEST_ASSERT(session.open(config));
    rund::replay::Binding binding{};
    std::size_t producers = 0u;
    auto first_source = [&](rund::replay::Writer &writer) {
      ++producers;
      Append(writer, std::array{std::byte{0x11}});
      return std::uint64_t{7u};
    };
    auto second_source = [&](rund::replay::Writer &writer) {
      ++producers;
      Append(writer, std::array{std::byte{0x22}});
      return std::uint64_t{9u};
    };
    const auto first = binding.input(
        rund::replay::Input{.id = 11u, .schema = 101u}, first_source);
    const auto second = binding.input(
        rund::replay::Input{.id = 12u, .schema = 102u}, second_source);
    const rund::replay::Record ordered =
        rund::replay::record(session, [&](rund::replay::Context &context) {
          const rund::replay::Value left = first.read(context);
          const rund::replay::Value right = second.read(context);
          TEST_ASSERT(left && left.sequence() == 7u);
          TEST_ASSERT(right && right.sequence() == 9u);
        });
    TEST_ASSERT(ordered);
    TEST_ASSERT(producers == 2u);
    const rund::replay::Check reproduced = rund::replay::run(
        session, ordered, [&](rund::replay::Context &context) {
          TEST_ASSERT(first.read(context).sequence() == 7u);
          TEST_ASSERT(second.read(context).sequence() == 9u);
        });
    TEST_ASSERT(reproduced);
    TEST_ASSERT(producers == 2u);
    rund::replay::Code wrong_code = rund::replay::Code::Ok;
    const rund::replay::Check wrong = rund::replay::run(
        session, ordered, [&](rund::replay::Context &context) {
          wrong_code = second.read(context).code();
        });
    TEST_ASSERT(!wrong);
    TEST_ASSERT(wrong_code == rund::replay::Code::InputOrderMismatch);
    TEST_ASSERT(producers == 2u);
    TEST_ASSERT(session.close());
  }

  config.replay.input_capacity = 1u;
  rund::Session replay;
  TEST_ASSERT(replay.open(config));
  std::size_t callback_calls = 0u;
  const rund::replay::Check rejected = rund::replay::run(
      replay, two, [&](rund::replay::Context &) { ++callback_calls; });
  TEST_ASSERT(!rejected);
  TEST_ASSERT(rejected.code() == rund::replay::Code::InputCapacityExceeded);
  TEST_ASSERT(callback_calls == 0u);
  TEST_ASSERT(replay.close());

  config.scheduler.host_event_capacity =
      std::numeric_limits<std::uint32_t>::max();
  config.replay.input_capacity = 1u;
  rund::Session impossible;
  const rund::Session::Status rejected_open = impossible.open(config);
  TEST_ASSERT(!rejected_open);
  TEST_ASSERT(rejected_open.code() ==
              rund::ReasonCode::HostReplayStorageInvalid);
  return 0;
}

} // namespace runtime_task_replay_run
