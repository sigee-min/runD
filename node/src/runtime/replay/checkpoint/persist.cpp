#include "local.hpp"

#include "../artifact/format.hpp"

namespace rund::replay {

Save Checkpoint::persist(const detail::Output output) const noexcept {
  if (!ok()) {
    return Save{code(), 0u, 0u};
  }
  const Data &data = *data_;
  node::replay_detail::artifact::Writer out{node::replay_detail::artifact::Sink{
      .state = output.state, .write = output.write}};
  static_cast<void>(
      out.header(node::replay_detail::artifact::Kind::Checkpoint) &&
      out.varuint(data.previous_segment_count) &&
      out.varuint(data.previous_input_position) &&
      out.fixed64(data.previous_prefix_hash) &&
      out.fixed64(data.previous_transcript_prefix_hash) &&
      out.fixed64(data.segment_record_hash) &&
      out.varuint(data.segment_input_count) &&
      out.fixed64(data.segment_input_hash) &&
      out.fixed64(data.segment_transcript_hash) &&
      out.varuint(data.state_schema) && out.varuint(data.state_size) &&
      out.fixed64(data.state_hash) && out.fixed64(data.checkpoint_hash) &&
      out.raw(data.state));
  const node::replay_detail::artifact::Result saved = out.finish();
  return Save{saved.code, saved.bytes, saved.writes};
}

} // namespace rund::replay
