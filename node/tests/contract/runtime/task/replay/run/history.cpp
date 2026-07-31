#include "local/model.hpp"

#include "test/assert.hpp"

#include <rund/host/env.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace runtime_task_replay_run {

int History(Model &model) {
  rund::replay::History history{rund::replay::Retention{
      .max_segments = 4u, .max_bytes = 1024u * 1024u, .max_events = 1024u}};
  const rund::replay::Append appended =
      model.binding.append(history, *model.baseline, model.first_state);
  TEST_ASSERT(appended);
  const std::optional<rund::replay::Segment> first =
      history.find(appended.sequence());
  TEST_ASSERT(first.has_value());
  TEST_ASSERT(!history.find(0u).has_value());
  TEST_ASSERT(!history.find(appended.sequence() + 1u).has_value());

  rund::replay::Checkpoint chain = first->checkpoint();
  std::uint64_t sequence = appended.sequence();
  for (std::size_t index = 0u; index < 5u; ++index) {
    const auto resume = model.binding.resume(chain);
    TEST_ASSERT(resume);
    const rund::replay::Record next =
        resume.record(model.session, [](rund::replay::Context &) noexcept {});
    TEST_ASSERT(next);
    const rund::replay::Append linked =
        model.binding.append(history, next, model.first_state);
    TEST_ASSERT(linked);
    TEST_ASSERT(linked.sequence() == sequence + 1u);
    sequence = linked.sequence();
    const std::optional<rund::replay::Segment> latest = history.latest();
    TEST_ASSERT(latest.has_value());
    chain = latest->checkpoint();
  }
  const rund::replay::Report retained = history.report();
  TEST_ASSERT(retained.retained_segments == 4u);
  TEST_ASSERT(retained.oldest_sequence == sequence - 3u);
  TEST_ASSERT(retained.newest_sequence == sequence);
  TEST_ASSERT(!history.find(retained.oldest_sequence - 1u).has_value());
  for (std::uint64_t retained_sequence = retained.oldest_sequence;
       retained_sequence <= retained.newest_sequence; ++retained_sequence) {
    const std::optional<rund::replay::Segment> segment =
        history.find(retained_sequence);
    TEST_ASSERT(segment.has_value());
    TEST_ASSERT(segment->sequence() == retained_sequence);
  }
  TEST_ASSERT(!history.find(retained.newest_sequence + 1u).has_value());

  const std::vector<std::byte> checkpoint_bytes =
      SaveReplayArtifact(first->checkpoint());
  const rund::replay::Load<rund::replay::Checkpoint> loaded =
      rund::replay::Checkpoint::load(checkpoint_bytes);
  TEST_ASSERT(loaded);
  TEST_ASSERT(loaded->hash() == first->checkpoint().hash());
  rund::replay::Load<rund::replay::Checkpoint> checkpoint_copy = loaded;
  rund::replay::Load<rund::replay::Checkpoint> checkpoint_moved =
      std::move(checkpoint_copy);
  TEST_ASSERT(checkpoint_moved);
  TEST_ASSERT(checkpoint_moved->hash() == loaded->hash());
  TEST_ASSERT(checkpoint_copy.code() ==
              rund::replay::Code::CheckpointLoadMovedFrom);
  constexpr std::array<std::byte, 1u> invalid{std::byte{0u}};
  TEST_ASSERT(!rund::replay::Checkpoint::load(invalid));
  rund::replay::Limits checkpoint_bytes_limit{};
  checkpoint_bytes_limit.max_bytes = checkpoint_bytes.size() - 1u;
  const auto checkpoint_bytes_rejected =
      rund::replay::Checkpoint::load(checkpoint_bytes, checkpoint_bytes_limit);
  TEST_ASSERT(!checkpoint_bytes_rejected);
  TEST_ASSERT(checkpoint_bytes_rejected.code() ==
              rund::replay::Code::CheckpointEncodedCapacityExceeded);
  rund::replay::Limits checkpoint_entries_limit{};
  checkpoint_entries_limit.max_entries = 0u;
  const auto checkpoint_entries_rejected = rund::replay::Checkpoint::load(
      checkpoint_bytes, checkpoint_entries_limit);
  TEST_ASSERT(!checkpoint_entries_rejected);
  TEST_ASSERT(checkpoint_entries_rejected.code() ==
              rund::replay::Code::CheckpointEntryCapacityExceeded);
  rund::replay::Limits checkpoint_state_limit{};
  checkpoint_state_limit.max_state_bytes = 0u;
  const auto checkpoint_state_rejected =
      rund::replay::Checkpoint::load(checkpoint_bytes, checkpoint_state_limit);
  TEST_ASSERT(!checkpoint_state_rejected);
  TEST_ASSERT(checkpoint_state_rejected.code() ==
              rund::replay::Code::CheckpointStateCapacityExceeded);

  const auto resume = model.binding.resume(first->checkpoint());
  TEST_ASSERT(resume);
  const std::size_t restores_before_invalid = model.restore_calls;
  const rund::replay::Binding wrong_binding{kStateSchema + 1u, model.restore};
  const auto invalid_resume = wrong_binding.resume(first->checkpoint());
  TEST_ASSERT(!invalid_resume);
  TEST_ASSERT(invalid_resume.code() == rund::replay::Code::StateSchemaInvalid);
  bool invalid_resume_callback = false;
  const rund::replay::Record invalid_resume_record =
      invalid_resume.record(model.session, [&](rund::replay::Context &) {
        invalid_resume_callback = true;
      });
  TEST_ASSERT(!invalid_resume_record);
  TEST_ASSERT(invalid_resume_record.code() ==
              rund::replay::Code::StateSchemaInvalid);
  TEST_ASSERT(!invalid_resume_callback);
  TEST_ASSERT(model.restore_calls == restores_before_invalid);

  const rund::replay::Record eventful =
      rund::replay::record(model.session, [](rund::replay::Context &) noexcept {
        (void)rund::host::env::get("RUND_REPLAY_SCOPE_EVENT_PROBE");
      });
  TEST_ASSERT(eventful);
  TEST_ASSERT(eventful.host_event_count() == 1u);

  std::size_t continuation_producers = 0u;
  auto continuation_source = [&](rund::replay::Writer &writer) {
    ++continuation_producers;
    Append(writer, model.payload);
    return std::uint64_t{1u};
  };
  const auto continuation_commands =
      model.binding.input(kInput, continuation_source);
  const rund::replay::Record continuation =
      resume.record(model.session, [&](rund::replay::Context &input) {
        const rund::replay::Value value = continuation_commands.read(input);
        TEST_ASSERT(value && value.sequence() == 1u);
      });
  TEST_ASSERT(continuation);
  TEST_ASSERT(continuation_producers == 1u);

  const std::array branch_bytes{std::byte{0x77}};
  const std::array branch_choice{
      continuation_commands.choice(1u, branch_bytes)};
  continuation_producers = 0u;
  const rund::replay::Scenario branch =
      resume.scenario(model.session, continuation,
                      std::span<const rund::replay::Choice>{branch_choice},
                      [&](rund::replay::Context &input) {
                        const rund::replay::Value value =
                            continuation_commands.read(input);
                        TEST_ASSERT(value && value.sequence() == 1u &&
                                    value.size() == branch_bytes.size());
                      });
  TEST_ASSERT(branch);
  TEST_ASSERT(continuation_producers == 0u);

  TEST_ASSERT(model.session.close());
  return 0;
}

} // namespace runtime_task_replay_run
