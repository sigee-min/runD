#include "model.hpp"

namespace package_blackbox::replay_detail {

[[nodiscard]] int CheckCodec(Fixture &fixture) {
  const rund::replay::Record &record = *fixture.record;
  const auto encoded = Persist(record);
  if (!encoded.saved) {
    return encoded.saved.exit_code();
  }
  const auto decoded = rund::replay::Record::load(encoded.bytes);
  if (!decoded) {
    return decoded.exit_code();
  }
  const rund::replay::Check codec_check = rund::replay::check(record, *decoded);
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
      diff.mismatch_count() != 0u || diff.mismatch(0u).has_value() || !window ||
      !window.error().empty() || window.exit_code() != 0 ||
      window.observation_index().has_value() ||
      window.host_event_index().has_value() ||
      window.input_index().has_value() || !window.expected_inputs().empty() ||
      !window.actual_inputs().empty() ||
      window.trace_record_index().has_value() ||
      record.observation_count() == 0u ||
      decoded->observation_count() != record.observation_count()) {
    return Mismatch("diff-window");
  }
  return 0;
}

} // namespace package_blackbox::replay_detail
