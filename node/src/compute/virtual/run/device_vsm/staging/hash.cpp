#include "hash.hpp"
#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

void increment(std::uint64_t &value) noexcept {
  if (value != std::numeric_limits<std::uint64_t>::max()) {
    ++value;
  }
}

} // namespace

bool reuse_output_hash(DeviceVsmProductRun &run) noexcept {
  if (run.owner == nullptr || run.projection == nullptr ||
      run.owner->evidence == nullptr || !run.owner->output_hash_valid ||
      run.input_count != run.owner->input_count ||
      run.input_count != run.projection->input_count) {
    return false;
  }
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    if (run.owner->output_hash_input_versions[index] !=
        run.projection->inputs[index].version) {
      return false;
    }
  }
  run.output_hash = run.owner->output_hash;
  increment(run.owner->output_hash_reuse_count);
  run.owner->evidence->output_hash_reuse_count =
      run.owner->output_hash_reuse_count;
  return true;
}

void observe_output_hash(DeviceVsmProductRun &run) noexcept {
  if (run.owner == nullptr || run.owner->evidence == nullptr) {
    return;
  }
  run.output_hash_observed = true;
  increment(run.owner->output_hash_observation_count);
  run.owner->evidence->output_hash_observation_count =
      run.owner->output_hash_observation_count;
}

void commit_output_hash(DeviceVsmProductRun &run) noexcept {
  if (!run.output_hash_observed || run.owner == nullptr ||
      run.projection == nullptr || run.input_count != run.owner->input_count ||
      run.input_count != run.projection->input_count) {
    return;
  }
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    run.owner->output_hash_input_versions[index] =
        run.projection->inputs[index].version;
  }
  run.owner->output_hash = run.output_hash;
  run.owner->output_hash_valid = true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
