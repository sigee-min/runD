#include "../batch.hpp"

#include "../../../../hash/fnv.hpp"

namespace rund::compute::detail {
namespace {

void mix(std::uint64_t &hash, const std::uint64_t value) noexcept {
  ::rund::node::hash_detail::MixU64(hash, value);
}

[[nodiscard]] Status publish_output_hash(PipelineState &state) noexcept {
  std::uint64_t hash = ::rund::node::hash_detail::kFnvOffset;
  mix(hash, state.outputs.size());
  for (const PipelineOutputState &observed : state.outputs) {
    if (!observed.observed || observed.resource >= state.resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineResource &value = state.resources[observed.resource];
    mix(hash, observed.resource);
    mix(hash, static_cast<std::uint64_t>(value.type));
    mix(hash, value.format.integer_bits);
    mix(hash, value.format.fraction_bits);
    mix(hash, static_cast<std::uint64_t>(value.format.rounding));
    mix(hash, static_cast<std::uint64_t>(value.format.overflow));
    mix(hash, static_cast<std::uint64_t>(value.format.approximation));
    mix(hash, value.count);
    mix(hash, observed.hash);
  }
  state.stats.output_hash = hash;
  return Status::success();
}

} // namespace

Status publish_pipeline_output_observation(PipelineState &state,
                                           const std::size_t output,
                                           const std::uint64_t hash) noexcept {
  if (output >= state.outputs.size()) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  PipelineOutputState &observed = state.outputs[output];
  if (observed.resource >= state.resources.size() ||
      state.resources[observed.resource].output != output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  observed.hash = hash;
  bool completed = false;
  if (!observed.observed && state.unobserved_outputs != 0u) {
    observed.observed = true;
    completed = --state.unobserved_outputs == 0u;
  }
  return completed ? publish_output_hash(state) : Status::success();
}

} // namespace rund::compute::detail
