#include "model.hpp"

namespace package_blackbox::replay_detail {

[[nodiscard]] int CheckRecord(Fixture &fixture) {
  rund::Session &replay_session = fixture.session;
  const auto &commands = fixture.commands;
  constexpr std::uint64_t sequence = Fixture::sequence;
  TaskRun recorded{};

  fixture.record.emplace(
      rund::replay::record(replay_session, [&](rund::replay::Context &context) {
        const auto value = commands.read(context);
        if (!value || value.sequence() != sequence || value.size() != 1u ||
            value.bytes()[0] != std::byte{0x2a}) {
          return;
        }
        RunBlackboxTask(recorded);
      }));
  const rund::replay::Record &record = *fixture.record;
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
      fixture.producers != 1u) {
    return Mismatch("record-check");
  }
  return 0;
}

} // namespace package_blackbox::replay_detail
