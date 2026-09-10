#include "model.hpp"

namespace package_blackbox::replay_detail {

[[nodiscard]] int CheckHistory(Fixture &fixture) {
  const rund::replay::Record &record = *fixture.record;
  const rund::replay::Record &continued = *fixture.continued;

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
  const rund::replay::Append rejected_segment = fixture.replay_binding.append(
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
      fixture.replay_binding.append(history, record, history_first_state);
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
  const std::uint64_t held_checkpoint_hash = held_segment->checkpoint().hash();
  const auto held_checkpoint_bytes = Persist(held_segment->checkpoint());
  if (!held_checkpoint_bytes.saved) {
    return held_checkpoint_bytes.saved.exit_code();
  }

  const rund::replay::Append history_next =
      fixture.replay_binding.append(history, continued, history_next_state);
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

  const auto resume = fixture.replay_binding.resume(*fixture.checkpoint);
  if (!resume) {
    return resume.exit_code();
  }
  bool continued_ran = false;
  auto replay_continuation = [&](rund::replay::Context &context) {
    continued_ran = true;
    fixture.ContinueSimulation(context);
  };
  const rund::replay::Check continued_replay = resume.run(
      fixture.session, current_segment->record(), replay_continuation);
  if (!continued_replay) {
    return continued_replay.exit_code();
  }
  if (!continued_ran || !fixture.continued_value_ok ||
      fixture.producers != 2u) {
    return Mismatch("runtime-continuation-replay");
  }

  bool checkpoint_scenario_value_ok = false;
  const std::array continued_bytes{std::byte{0x2c}};
  const std::array continued_choices{
      fixture.commands.choice(Fixture::continued_sequence, continued_bytes)};
  const rund::replay::Scenario continued_scenario =
      resume.scenario(fixture.session, continued, continued_choices,
                      [&](rund::replay::Context &context) {
                        const auto value = fixture.commands.read(context);
                        checkpoint_scenario_value_ok =
                            value &&
                            value.sequence() == Fixture::continued_sequence &&
                            value.bytes()[0] == std::byte{0x2c};
                      });
  if (!continued_scenario) {
    return continued_scenario.exit_code();
  }
  return continued_scenario && continued_scenario.callback_ran() &&
                 fixture.producers == 2u && checkpoint_scenario_value_ok &&
                 continued_scenario.actual().has_value() &&
                 continued_scenario.diff().has_value() &&
                 !continued_scenario.matches()
             ? 0
             : Mismatch("runtime-continuation-scenario");
}

} // namespace package_blackbox::replay_detail
