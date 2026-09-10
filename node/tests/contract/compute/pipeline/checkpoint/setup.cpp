#include "local.hpp"

namespace rund_node_test_pipeline::checkpoint {

bool SameCheckpointStats(const rund::compute::CheckpointStats &left,
                         const rund::compute::CheckpointStats &right) noexcept {
  return left.device_state_acquire_count == right.device_state_acquire_count &&
         left.device_state_rebase_count == right.device_state_rebase_count &&
         left.device_state_copy_byte_count ==
             right.device_state_copy_byte_count &&
         left.device_state_copy_command_count ==
             right.device_state_copy_command_count &&
         left.reusable_snapshot_count == right.reusable_snapshot_count &&
         left.reusable_snapshot_byte_count ==
             right.reusable_snapshot_byte_count &&
         left.reusable_snapshot_hash == right.reusable_snapshot_hash &&
         left.reusable_snapshot_transfer_count ==
             right.reusable_snapshot_transfer_count;
}

Preparation Prepare(rund::compute::Device &device, const Backend backend) {
  using namespace rund::compute;
  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-reusable-checkpoint", Initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = Upload(device, Initial);
  auto second = device.buffer<std::int32_t>(Initial.size());
  if (!advance || !first || !second) {
    return {.error = 1};
  }
  auto prepared = pipeline(device)
                      .state(*first, *second)
                      .then(*advance, read(*first), write(*second))
                      .commit()
                      .prepare();
  if (!prepared) {
    return {.error = 2};
  }
  std::shared_ptr<detail::PipelineState> source_state =
      detail::PipelineStateAccess::state(*prepared);
  if (source_state == nullptr || source_state->publication == nullptr) {
    return {.error = 3};
  }

  auto storage = prepared->snapshot_storage();
  auto small = prepared->snapshot_storage(sizeof(Initial) - 1u);
  auto latest = prepared->latest_device_state();
  if (!storage || !small || !latest) {
    return {.error = 4};
  }
  return {
      .context =
          Context{
              device,
              backend,
              std::move(*advance),
              std::move(*first),
              std::move(*second),
              std::move(*prepared),
              std::move(source_state),
              std::move(*storage),
              std::move(*small),
              std::move(*latest),
          },
  };
}

} // namespace rund_node_test_pipeline::checkpoint
