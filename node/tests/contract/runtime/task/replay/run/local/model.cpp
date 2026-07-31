#include "model.hpp"

#include "test/assert.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace runtime_task_replay_run {

void Append(rund::replay::Writer &writer,
            const std::span<const std::byte> bytes) {
  if (!writer.append(bytes)) {
    throw std::runtime_error{std::string{writer.error()}};
  }
}

Model::Model(rund::SessionConfig initial)
    : config{std::move(initial)}, restore{*this},
      binding{kStateSchema, restore}, source{*this},
      commands{binding.input(kInput, source)}, step{*this} {}

rund::replay::Restore
Model::Restore::operator()(const std::span<const std::byte> state) const {
  ++model.restore_calls;
  TEST_ASSERT(state.size() == model.first_state.size());
  return rund::replay::Restore::Restored;
}

std::uint64_t Model::Source::operator()(rund::replay::Writer &writer) const {
  ++model.producer_calls;
  Append(writer, model.payload);
  return kSequence;
}

void Model::Step::operator()(rund::replay::Context &input,
                             rund::Session &active) const {
  TEST_ASSERT(&active == &model.session);
  ++model.callback_calls;
  const rund::replay::Value value = model.commands.read(input);
  TEST_ASSERT(value);
  TEST_ASSERT(value.sequence() == kSequence);
  model.observed_size = value.size();
}

} // namespace runtime_task_replay_run
