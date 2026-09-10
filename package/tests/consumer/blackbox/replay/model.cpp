#include "model.hpp"

namespace package_blackbox::replay_detail {

rund::replay::Restore
Restore::operator()(const std::span<const std::byte> bytes) const {
  fixture->restored_before_record = bytes.size() == 2u &&
                                    bytes[0] == std::byte{0x51} &&
                                    bytes[1] == std::byte{0x52};
  return fixture->restored_before_record ? rund::replay::Restore::Restored
                                         : rund::replay::Restore::Failed;
}

std::uint64_t Source::operator()(rund::replay::Writer &writer) const {
  ++fixture->producers;
  const std::array bytes{fixture->source_value};
  (void)writer.append(bytes);
  return fixture->source_sequence;
}

Fixture::Fixture(rund::Session &session)
    : session(session), state{std::byte{0x51}, std::byte{0x52}}, restore{this},
      replay_binding(state_schema, restore), source{this},
      commands(replay_binding.input(input, source)) {}

void Fixture::ContinueSimulation(rund::replay::Context &context) {
  continuation_record_ran = restored_before_record;
  const auto value = commands.read(context);
  continued_value_ok = value && value.sequence() == continued_sequence &&
                       value.size() == 1u &&
                       value.bytes()[0] == continued_expected;
}

} // namespace package_blackbox::replay_detail
