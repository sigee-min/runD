#include "model.hpp"

namespace package_blackbox::replay_detail {

[[nodiscard]] int CheckCheckpoint(Fixture &fixture) {
  const rund::replay::Record &record = *fixture.record;
  const rund::replay::Checkpoint checkpoint =
      fixture.replay_binding.checkpoint(record, fixture.state);
  if (!checkpoint) {
    return checkpoint.exit_code();
  }
  if (checkpoint.segment_count() != 1u ||
      checkpoint.input_position() != record.input_count() ||
      checkpoint.schema() != Fixture::state_schema ||
      checkpoint.state_size() != 2u || checkpoint.state_hash() == 0u ||
      checkpoint.boundary_hash() == 0u || checkpoint.prefix_hash() == 0u ||
      checkpoint.transcript_prefix_hash() == 0u || checkpoint.hash() == 0u) {
    return Mismatch("runtime-checkpoint");
  }
  const std::uint64_t checkpoint_state_hash = checkpoint.state_hash();
  const std::uint64_t checkpoint_hash = checkpoint.hash();
  const auto checkpoint_bytes = Persist(checkpoint);
  if (!checkpoint_bytes.saved) {
    return checkpoint_bytes.saved.exit_code();
  }
  fixture.state[0] = std::byte{0xff};
  fixture.state[1] = std::byte{0xee};
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
  fixture.checkpoint.emplace(*loaded);
  const rund::replay::Checkpoint &persisted_checkpoint = *loaded;

  const auto resume = fixture.replay_binding.resume(persisted_checkpoint);
  if (!resume) {
    return resume.exit_code();
  }
  fixture.source_sequence = Fixture::continued_sequence;
  fixture.source_value = Fixture::continued_expected;
  const rund::replay::Record continued =
      resume.record(fixture.session, [&](rund::replay::Context &context) {
        fixture.ContinueSimulation(context);
      });
  if (!continued) {
    return continued.exit_code();
  }
  if (!fixture.continuation_record_ran || !fixture.continued_value_ok ||
      fixture.producers != 2u) {
    return Mismatch("runtime-continuation-record");
  }
  fixture.continued.emplace(continued);

  const rund::replay::Check mismatched = rund::replay::check(record, continued);
  if (mismatched ||
      mismatched.code() != rund::replay::Code::RecordStartMismatch ||
      mismatched.error().empty() || mismatched.exit_code() != 1) {
    return Mismatch("runtime-check-mismatch");
  }
  return 0;
}

} // namespace package_blackbox::replay_detail
