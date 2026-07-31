#include "local/model.hpp"

#include "test/assert.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace runtime_task_replay_run {

int Scenario(Model &model) {
  model.producer_calls = 0u;
  model.callback_calls = 0u;
  const rund::replay::Check replayed =
      rund::replay::run(model.session, *model.baseline, model.step);
  TEST_ASSERT(replayed);
  TEST_ASSERT(model.producer_calls == 0u);
  TEST_ASSERT(model.callback_calls == 1u);
  TEST_ASSERT(replayed.actual_hash() == model.baseline->hash());

  const std::array shorter{std::byte{0x41}};
  const std::array longer{std::byte{0x41}, std::byte{0x42}, std::byte{0x43},
                          std::byte{0x44}, std::byte{0x45}};
  const std::array<std::span<const std::byte>, 3u> replacements{
      std::span<const std::byte>{shorter}, std::span<const std::byte>{longer},
      std::span<const std::byte>{}};
  for (const std::span<const std::byte> replacement : replacements) {
    const std::array choices{model.commands.choice(kSequence, replacement)};
    model.producer_calls = 0u;
    model.callback_calls = 0u;
    const rund::replay::Scenario changed = rund::replay::scenario(
        model.session, *model.baseline,
        std::span<const rund::replay::Choice>{choices}, model.step);
    TEST_ASSERT(changed);
    TEST_ASSERT(changed.callback_ran());
    TEST_ASSERT(model.producer_calls == 0u);
    TEST_ASSERT(model.callback_calls == 1u);
    TEST_ASSERT(model.observed_size == replacement.size());
    TEST_ASSERT(!changed.matches());
  }

  bool invalid_callback = false;
  const std::array duplicate_bytes{std::byte{0x31}};
  const rund::replay::Choice duplicate =
      model.commands.choice(kSequence, duplicate_bytes);
  const std::array duplicates{duplicate, duplicate};
  const rund::replay::Scenario rejected = rund::replay::scenario(
      model.session, *model.baseline,
      std::span<const rund::replay::Choice>{duplicates},
      [&](rund::replay::Context &) { invalid_callback = true; });
  TEST_ASSERT(!rejected);
  TEST_ASSERT(rejected.code() == rund::replay::Code::ScenarioInputDuplicate);
  TEST_ASSERT(!invalid_callback);

  std::size_t invalid_bind_producers = 0u;
  bool bind_error_visible = false;
  auto invalid_source = [&](rund::replay::Writer &writer) {
    ++invalid_bind_producers;
    Append(writer, model.payload);
    return kSequence;
  };
  const auto invalid_channel = model.binding.input(
      rund::replay::Input{.id = 0u, .schema = kInput.schema}, invalid_source);
  const rund::replay::Record invalid_bind =
      rund::replay::record(model.session, [&](rund::replay::Context &input) {
        const rund::replay::Value value = invalid_channel.read(input);
        bind_error_visible =
            !value && value.code() == rund::replay::Code::InputIdInvalid;
      });
  TEST_ASSERT(!invalid_bind);
  TEST_ASSERT(bind_error_visible);
  TEST_ASSERT(invalid_bind_producers == 0u);

  auto uncommitted_source = [](rund::replay::Writer &) { return kSequence; };
  const auto uncommitted_channel =
      model.binding.input(kInput, uncommitted_source);
  const rund::replay::Record uncommitted =
      rund::replay::record(model.session, [&](rund::replay::Context &input) {
        const rund::replay::Value ignored = uncommitted_channel.read(input);
        (void)ignored;
      });
  TEST_ASSERT(!uncommitted);

  std::size_t empty_producer_calls = 0u;
  auto empty_source = [&](rund::replay::Writer &writer) {
    ++empty_producer_calls;
    const std::span<std::byte> acquired = writer.acquire(0u);
    TEST_ASSERT(acquired.empty());
    TEST_ASSERT(writer.commit(0u));
    return kSequence;
  };
  const auto empty_channel = model.binding.input(kInput, empty_source);
  const rund::replay::Record empty =
      rund::replay::record(model.session, [&](rund::replay::Context &input) {
        const rund::replay::Value value = empty_channel.read(input);
        TEST_ASSERT(value);
        TEST_ASSERT(value.bytes().empty());
      });
  TEST_ASSERT(empty);
  TEST_ASSERT(empty_producer_calls == 1u);

  const std::array chunk_first{std::byte{0x11}, std::byte{0x22}};
  const std::array chunk_second{std::byte{0x33}};
  const std::array chunk_third{std::byte{0x44}, std::byte{0x55}};
  const std::array combined{std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
                            std::byte{0x44}, std::byte{0x55}};
  auto chunk_source = [&](rund::replay::Writer &writer) {
    TEST_ASSERT(writer.append(chunk_first));
    TEST_ASSERT(writer.append(chunk_second));
    std::span<std::byte> tail = writer.acquire(chunk_third.size());
    TEST_ASSERT(tail.size() == chunk_third.size());
    std::copy(chunk_third.begin(), chunk_third.end(), tail.begin());
    TEST_ASSERT(writer.commit(chunk_third.size()));
    TEST_ASSERT(writer.size() == combined.size());
    return std::uint64_t{8u};
  };
  const auto chunk_channel = model.binding.input(kInput, chunk_source);
  const rund::replay::Record chunked =
      rund::replay::record(model.session, [&](rund::replay::Context &input) {
        const rund::replay::Value value = chunk_channel.read(input);
        TEST_ASSERT(value);
        TEST_ASSERT(std::equal(value.bytes().begin(), value.bytes().end(),
                               combined.begin(), combined.end()));
      });
  TEST_ASSERT(chunked);

  bool healthy_after_failure = false;
  TEST_ASSERT(
      rund::replay::live(model.session, [&](rund::replay::Context &input) {
        const rund::replay::Value value = model.commands.read(input);
        healthy_after_failure = value.ok();
      }));
  TEST_ASSERT(healthy_after_failure);
  return 0;
}

} // namespace runtime_task_replay_run
