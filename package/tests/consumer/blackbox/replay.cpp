#include "model.hpp"

namespace package_blackbox {

[[nodiscard]] int CheckRunReplay() {
  const rund::SessionConfig config{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
              .observation_capacity = 8u,
          },
  };
  TaskRun run_task{};
  const rund::Session::Result result =
      rund::run(config, [&] { RunBlackboxTask(run_task); });
  if (!result) {
    return result.exit_code();
  }
  if (!run_task.joined) {
    return run_task.joined.exit_code();
  }
  if (!run_task.operation) {
    return run_task.operation.exit_code();
  }
  if (!result.ok() || run_task.ran != 1u || result.tasks().spawned() != 1u ||
      result.tasks().completed() != 1u) {
    return Mismatch("run");
  }

  rund::Session replay_session{};
  const auto replay_opened = replay_session.open(config);
  if (!replay_opened) {
    return replay_opened.exit_code();
  }

  return Finish(replay_session, [&]() -> int {
    constexpr rund::replay::Input input{.id = 17u, .schema = 401u};
    constexpr std::uint64_t sequence = 29u;
    std::vector<std::byte> state{std::byte{0x51}, std::byte{0x52}};
    std::byte *const state_data = state.data();
    constexpr std::uint64_t state_schema = 701u;
    bool restored_before_record = false;
    bool continuation_record_ran = false;
    auto restore = [&](const std::span<const std::byte> bytes) {
      restored_before_record = bytes.size() == 2u &&
                               bytes[0] == std::byte{0x51} &&
                               bytes[1] == std::byte{0x52};
      return restored_before_record ? rund::replay::Restore::Restored
                                    : rund::replay::Restore::Failed;
    };
    rund::replay::Binding replay_binding{state_schema, restore};
    std::uint64_t source_sequence = sequence;
    std::byte source_value = std::byte{0x2a};
    TaskRun recorded{};
    std::uint32_t producers = 0u;
    auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
      ++producers;
      const std::array bytes{source_value};
      (void)writer.append(bytes);
      return source_sequence;
    };
    const auto commands = replay_binding.input(input, source);
    const rund::replay::Record record = rund::replay::record(
        replay_session, [&](rund::replay::Context &context) {
          const auto value = commands.read(context);
          if (!value || value.sequence() != sequence || value.size() != 1u ||
              value.bytes()[0] != std::byte{0x2a}) {
            return;
          }
          RunBlackboxTask(recorded);
        });
    if (!record) {
      return record.exit_code();
    }
    if (!recorded.joined) {
      return recorded.joined.exit_code();
    }
    if (!recorded.operation) {
      return recorded.operation.exit_code();
    }
    const rund::replay::Check replay = rund::replay::check(record, record);
    if (!replay) {
      return replay.exit_code();
    }
    if (!record.error().empty() || record.exit_code() != 0 ||
        record.input_count() != 1u || record.input_hash() == 0u ||
        !record.captures().empty() || record.capture_hash() != 0u ||
        record.capture_report().retained_records != 0u ||
        record.transcript_hash() == 0u || record.hash() == 0u ||
        !replay.error().empty() || replay.exit_code() != 0 ||
        replay.expected_hash() != replay.actual_hash() ||
        !replay.actual().has_value() ||
        replay.actual_hash() != replay.actual()->hash() ||
        replay.actual()->tasks().completed() == 0u || recorded.ran != 1u ||
        producers != 1u) {
      return Mismatch("record-check");
    }

    const auto encoded = Persist(record);
    if (!encoded.saved) {
      return encoded.saved.exit_code();
    }
    const auto decoded = rund::replay::Record::load(encoded.bytes);
    if (!decoded) {
      return decoded.exit_code();
    }
    const rund::replay::Check codec_check =
        rund::replay::check(record, *decoded);
    if (!codec_check) {
      return codec_check.exit_code();
    }
    if (!decoded.ok() || !decoded.error().empty() || decoded.exit_code() != 0) {
      return Mismatch("codec-check");
    }
    constexpr std::string_view invalid_text = "invalid";
    const auto rejected = rund::replay::Record::load(
        std::as_bytes(std::span{invalid_text.data(), invalid_text.size()}));
    if (rejected.ok() || rejected || rejected.error().empty() ||
        rejected.exit_code() != 1) {
      return Mismatch("codec-error");
    }

    const rund::replay::Diff diff = rund::replay::diff(record, *decoded);
    const rund::replay::Window window =
        rund::replay::window(record, *decoded, 1u);
    if (!diff) {
      return diff.exit_code();
    }
    if (!window) {
      return window.exit_code();
    }
    if (!diff.error().empty() || diff.exit_code() != 0 ||
        diff.mismatch_count() != 0u || diff.mismatch(0u).has_value() ||
        !window || !window.error().empty() || window.exit_code() != 0 ||
        window.observation_index().has_value() ||
        window.host_event_index().has_value() ||
        window.input_index().has_value() || !window.expected_inputs().empty() ||
        !window.actual_inputs().empty() ||
        window.trace_record_index().has_value() ||
        record.observation_count() == 0u ||
        decoded->observation_count() != record.observation_count()) {
      return Mismatch("diff-window");
    }

    TaskRun replayed{};
    source_value = std::byte{0xff};
    const rund::replay::Check replayed_check = rund::replay::run(
        replay_session, record, [&](rund::replay::Context &context) {
          const auto value = commands.read(context);
          if (!value || value.sequence() != sequence || value.size() != 1u ||
              value.bytes()[0] != std::byte{0x2a}) {
            return;
          }
          RunBlackboxTask(replayed);
        });
    if (!replayed_check) {
      return replayed_check.exit_code();
    }
    if (!replayed.joined) {
      return replayed.joined.exit_code();
    }
    if (!replayed.operation) {
      return replayed.operation.exit_code();
    }
    if (!replayed_check.actual().has_value() ||
        replayed_check.actual()->tasks().completed() == 0u ||
        replayed.ran != 1u || producers != 1u) {
      return Mismatch("runtime-replay");
    }

    TaskRun scenario_run{};
    bool scenario_value_ok = false;
    const std::array choice_bytes{std::byte{0x2b}};
    const std::array choices{commands.choice(sequence, choice_bytes)};
    const rund::replay::Scenario scenario = rund::replay::scenario(
        replay_session, record, choices, [&](rund::replay::Context &context) {
          const auto value = commands.read(context);
          scenario_value_ok = value && value.sequence() == sequence &&
                              value.bytes()[0] == std::byte{0x2b};
          RunBlackboxTask(scenario_run);
        });
    if (!scenario) {
      return scenario.exit_code();
    }
    if (!scenario_run.joined) {
      return scenario_run.joined.exit_code();
    }
    if (!scenario_run.operation) {
      return scenario_run.operation.exit_code();
    }
    if (!scenario.callback_ran() || producers != 1u || !scenario_value_ok ||
        scenario_run.ran != 1u || !scenario.actual().has_value() ||
        !scenario.diff().has_value() || scenario.matches()) {
      return Mismatch("runtime-scenario");
    }
    const rund::replay::Window changed =
        rund::replay::window(record, *scenario.actual(), 1u);
    const auto expected_inputs = changed.expected_inputs();
    const auto actual_inputs = changed.actual_inputs();
    if (changed || changed.code() != rund::replay::Code::InputHashMismatch ||
        changed.input_index() != 0u || expected_inputs.size() != 1u ||
        actual_inputs.size() != 1u || expected_inputs.front().input != input ||
        actual_inputs.front().input != input ||
        expected_inputs.front().sequence != sequence ||
        actual_inputs.front().sequence != sequence ||
        expected_inputs.front().index != 0u ||
        actual_inputs.front().index != 0u ||
        expected_inputs.front().size != 1u ||
        actual_inputs.front().size != 1u ||
        expected_inputs.front().hash == actual_inputs.front().hash) {
      return Mismatch("runtime-scenario-window");
    }

    std::uint32_t rejected_callbacks = 0u;
    auto reject_callback = [&](rund::replay::Context &) {
      ++rejected_callbacks;
    };
    const std::array duplicate_first{std::byte{0x2b}};
    const std::array duplicate_second{std::byte{0x2c}};
    const std::array duplicate_choices{
        commands.choice(sequence, duplicate_first),
        commands.choice(sequence, duplicate_second),
    };
    const rund::replay::Scenario duplicate = rund::replay::scenario(
        replay_session, record, duplicate_choices, reject_callback);
    if (duplicate ||
        duplicate.code() != rund::replay::Code::ScenarioInputDuplicate ||
        duplicate.error().empty() || duplicate.exit_code() != 1 ||
        duplicate.callback_ran() || rejected_callbacks != 0u) {
      return Mismatch("runtime-scenario-duplicate");
    }

    const std::array missing_bytes{std::byte{0x2b}};
    const std::array missing_choices{
        commands.choice(sequence + 1u, missing_bytes)};
    const rund::replay::Scenario missing = rund::replay::scenario(
        replay_session, record, missing_choices, reject_callback);
    if (missing || missing.code() != rund::replay::Code::ScenarioInputMissing ||
        missing.error().empty() || missing.exit_code() != 1 ||
        missing.callback_ran() || rejected_callbacks != 0u) {
      return Mismatch("runtime-scenario-missing");
    }

    const rund::replay::Checkpoint checkpoint =
        replay_binding.checkpoint(record, state);
    if (!checkpoint) {
      return checkpoint.exit_code();
    }
    if (checkpoint.segment_count() != 1u ||
        checkpoint.input_position() != record.input_count() ||
        checkpoint.schema() != state_schema || checkpoint.state_size() != 2u ||
        checkpoint.state_hash() == 0u || checkpoint.boundary_hash() == 0u ||
        checkpoint.prefix_hash() == 0u ||
        checkpoint.transcript_prefix_hash() == 0u || checkpoint.hash() == 0u) {
      return Mismatch("runtime-checkpoint");
    }
    const std::uint64_t checkpoint_state_hash = checkpoint.state_hash();
    const std::uint64_t checkpoint_hash = checkpoint.hash();
    const auto checkpoint_bytes = Persist(checkpoint);
    if (!checkpoint_bytes.saved) {
      return checkpoint_bytes.saved.exit_code();
    }
    state_data[0] = std::byte{0xff};
    state_data[1] = std::byte{0xee};
    const Artifact checkpoint_copy = Persist(checkpoint);
    if (!checkpoint_copy.saved) {
      return checkpoint_copy.saved.exit_code();
    }
    if (checkpoint.state()[0] != std::byte{0x51} ||
        checkpoint.state()[1] != std::byte{0x52} ||
        checkpoint.state_hash() != checkpoint_state_hash ||
        checkpoint.hash() != checkpoint_hash ||
        checkpoint_copy.bytes != checkpoint_bytes.bytes) {
      return Mismatch("runtime-checkpoint-snapshot");
    }
    const auto loaded = rund::replay::Checkpoint::load(
        checkpoint_bytes.bytes,
        rund::replay::Limits{.max_state_bytes = checkpoint.state_size()});
    if (!loaded) {
      return loaded.exit_code();
    }
    const Artifact loaded_copy = Persist(*loaded);
    if (!loaded_copy.saved) {
      return loaded_copy.saved.exit_code();
    }
    if (loaded->hash() != checkpoint.hash() ||
        loaded->state_size() != checkpoint.state_size() ||
        loaded->state()[0] != std::byte{0x51} ||
        loaded->state()[1] != std::byte{0x52} ||
        loaded_copy.bytes != checkpoint_bytes.bytes) {
      return Mismatch("runtime-checkpoint-codec");
    }
    const rund::replay::Checkpoint &persisted_checkpoint = *loaded;

    const auto resume = replay_binding.resume(persisted_checkpoint);
    if (!resume) {
      return resume.exit_code();
    }
    constexpr std::uint64_t continued_sequence = 30u;
    std::byte continued_expected = std::byte{0x2d};
    bool continued_value_ok = false;
    source_sequence = continued_sequence;
    source_value = continued_expected;
    auto continue_simulation = [&](rund::replay::Context &context) {
      continuation_record_ran = restored_before_record;
      const auto value = commands.read(context);
      continued_value_ok = value && value.sequence() == continued_sequence &&
                           value.size() == 1u &&
                           value.bytes()[0] == continued_expected;
    };
    const rund::replay::Record continued =
        resume.record(replay_session, continue_simulation);
    if (!continued) {
      return continued.exit_code();
    }
    if (!continuation_record_ran || !continued_value_ok || producers != 2u) {
      return Mismatch("runtime-continuation-record");
    }

    const rund::replay::Check mismatched =
        rund::replay::check(record, continued);
    if (mismatched ||
        mismatched.code() != rund::replay::Code::RecordStartMismatch ||
        mismatched.error().empty() || mismatched.exit_code() != 1) {
      return Mismatch("runtime-check-mismatch");
    }

    const rund::replay::History invalid_history{rund::replay::Retention{
        .max_segments = 0u,
        .max_bytes = 1u,
        .max_events = 1u,
    }};
    if (invalid_history ||
        invalid_history.code() != rund::replay::Code::RetentionInvalid ||
        invalid_history.error().empty() || invalid_history.exit_code() != 1) {
      return Mismatch("runtime-history-invalid");
    }

    rund::replay::History bounded_history{rund::replay::Retention{
        .max_segments = 1u,
        .max_bytes = 1u,
        .max_events = 1u,
    }};
    const rund::replay::Append rejected_segment = replay_binding.append(
        bounded_history, record, std::span<const std::byte>{});
    if (!bounded_history) {
      return bounded_history.exit_code();
    }
    if (rejected_segment ||
        rejected_segment.code() !=
            rund::replay::Code::RetentionSegmentExceedsBounds ||
        rejected_segment.error().empty() || rejected_segment.exit_code() != 1) {
      return Mismatch("runtime-history-capacity");
    }

    constexpr rund::replay::Retention history_bounds{
        .max_segments = 1u,
        .max_bytes = 1024u * 1024u * 1024u,
        .max_events = 1024u * 1024u,
    };
    const std::array history_first_state{std::byte{0x51}, std::byte{0x52}};
    const std::array history_next_state{std::byte{0x53}, std::byte{0x54}};
    rund::replay::History history{history_bounds};
    const rund::replay::Append history_first =
        replay_binding.append(history, record, history_first_state);
    if (!history) {
      return history.exit_code();
    }
    if (!history_first) {
      return history_first.exit_code();
    }
    const auto held_segment = history.find(history_first.sequence());
    if (history_first.evicted_segments() != 0u || !held_segment.has_value()) {
      return Mismatch("runtime-history-first");
    }
    const std::uint64_t held_record_hash = held_segment->record().hash();
    const std::uint64_t held_checkpoint_hash =
        held_segment->checkpoint().hash();
    const auto held_checkpoint_bytes = Persist(held_segment->checkpoint());
    if (!held_checkpoint_bytes.saved) {
      return held_checkpoint_bytes.saved.exit_code();
    }

    const rund::replay::Append history_next =
        replay_binding.append(history, continued, history_next_state);
    if (!history_next) {
      return history_next.exit_code();
    }
    const auto current_segment = history.find(history_next.sequence());
    const rund::replay::Report history_report = history.report();
    const Artifact held_checkpoint_copy = Persist(held_segment->checkpoint());
    if (!held_checkpoint_copy.saved) {
      return held_checkpoint_copy.saved.exit_code();
    }
    if (history_next.evicted_segments() != 1u || history.size() != 1u ||
        history.find(history_first.sequence()).has_value() ||
        !current_segment.has_value() ||
        held_segment->record().hash() != held_record_hash ||
        held_segment->checkpoint().hash() != held_checkpoint_hash ||
        held_checkpoint_copy.bytes != held_checkpoint_bytes.bytes ||
        held_segment->checkpoint().state()[0] != std::byte{0x51} ||
        held_segment->checkpoint().state()[1] != std::byte{0x52} ||
        history_report.retained_segments != 1u ||
        history_report.appended_segments != 2u ||
        history_report.evicted_segments != 1u ||
        history_report.rejected_segments != 0u ||
        history_report.oldest_sequence != history_next.sequence() ||
        history_report.newest_sequence != history_next.sequence() ||
        history_report.retained_bytes != current_segment->byte_count() ||
        history_report.retained_events != current_segment->event_count() ||
        history_report.retained_bytes > history_bounds.max_bytes ||
        history_report.retained_events > history_bounds.max_events ||
        history_report.prefix_hash !=
            current_segment->checkpoint().prefix_hash() ||
        history_report.transcript_prefix_hash !=
            current_segment->checkpoint().transcript_prefix_hash()) {
      return Mismatch("runtime-history-retention");
    }

    const rund::replay::Diff history_diff =
        rund::replay::diff(continued, current_segment->record());
    const rund::replay::Window history_window =
        rund::replay::window(continued, current_segment->record(), 1u);
    if (!history_diff) {
      return history_diff.exit_code();
    }
    if (!history_window) {
      return history_window.exit_code();
    }
    if (history_diff.mismatch_count() != 0u ||
        history_window.observation_index().has_value() ||
        history_window.host_event_index().has_value() ||
        history_window.trace_record_index().has_value()) {
      return Mismatch("runtime-history-evidence");
    }

    bool continued_ran = false;
    auto replay_continuation = [&](rund::replay::Context &context) {
      continued_ran = true;
      continue_simulation(context);
    };
    const rund::replay::Check continued_replay = resume.run(
        replay_session, current_segment->record(), replay_continuation);
    if (!continued_replay) {
      return continued_replay.exit_code();
    }
    if (!continued_ran || !continued_value_ok || producers != 2u) {
      return Mismatch("runtime-continuation-replay");
    }

    bool checkpoint_scenario_value_ok = false;
    const std::array continued_bytes{std::byte{0x2c}};
    const std::array continued_choices{
        commands.choice(continued_sequence, continued_bytes)};
    const rund::replay::Scenario continued_scenario =
        resume.scenario(replay_session, continued, continued_choices,
                        [&](rund::replay::Context &context) {
                          const auto value = commands.read(context);
                          checkpoint_scenario_value_ok =
                              value && value.sequence() == continued_sequence &&
                              value.bytes()[0] == std::byte{0x2c};
                        });
    if (!continued_scenario) {
      return continued_scenario.exit_code();
    }
    return continued_scenario && continued_scenario.callback_ran() &&
                   producers == 2u && checkpoint_scenario_value_ok &&
                   continued_scenario.actual().has_value() &&
                   continued_scenario.diff().has_value() &&
                   !continued_scenario.matches()
               ? 0
               : Mismatch("runtime-continuation-scenario");
  });
}

} // namespace package_blackbox
