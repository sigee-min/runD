#include "model.hpp"

namespace package_blackbox::replay_detail {

[[nodiscard]] int CheckScenario(Fixture &fixture) {
  rund::Session &replay_session = fixture.session;
  const auto &commands = fixture.commands;
  const rund::replay::Record &record = *fixture.record;
  constexpr rund::replay::Input input = Fixture::input;
  constexpr std::uint64_t sequence = Fixture::sequence;

  TaskRun replayed{};
  fixture.source_value = std::byte{0xff};
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
      replayed.ran != 1u || fixture.producers != 1u) {
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
  if (!scenario.callback_ran() || fixture.producers != 1u ||
      !scenario_value_ok || scenario_run.ran != 1u ||
      !scenario.actual().has_value() || !scenario.diff().has_value() ||
      scenario.matches()) {
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
      actual_inputs.front().index != 0u || expected_inputs.front().size != 1u ||
      actual_inputs.front().size != 1u ||
      expected_inputs.front().hash == actual_inputs.front().hash) {
    return Mismatch("runtime-scenario-window");
  }

  std::uint32_t rejected_callbacks = 0u;
  auto reject_callback = [&](rund::replay::Context &) { ++rejected_callbacks; };
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
  return 0;
}

} // namespace package_blackbox::replay_detail
