#include "prepare/local.hpp"

namespace rund::compute::detail::virtual_run_overlap {
namespace {

[[nodiscard]] Status prepare_epoch(prepare_detail::Context &context) noexcept {
  Status status = prepare_detail::admit(context);
  if (!status) {
    return status;
  }
  status = prepare_detail::supply(context);
  if (!status) {
    return status;
  }
  return prepare_detail::schedule_lookahead(context);
}

} // namespace

Status prepare_cpu_epoch(VirtualPipelineState &state, VirtualBacking &input,
                         const VirtualRunProjection &run,
                         const std::uint64_t epoch,
                         std::array<bool, 2u> &prefetch_pending, Stats &stats,
                         PreparedEpoch &prepared, TimelineInterval &ready,
                         TimelineInterval &upload,
                         const bool coherent_lookahead, bool &poison,
                         VirtualInputReuseSeed *const input_reuse) noexcept {
  prepare_detail::Context context{state,
                                  input,
                                  run,
                                  epoch,
                                  prefetch_pending,
                                  stats,
                                  prepared,
                                  ready,
                                  upload,
                                  coherent_lookahead,
                                  poison,
                                  input_reuse,
                                  prepare_detail::Backend::Cpu};
  return prepare_epoch(context);
}

Status prepare_accel_epoch(VirtualPipelineState &state, VirtualBacking &input,
                           const VirtualRunProjection &run,
                           const std::uint64_t epoch,
                           std::array<bool, 2u> &prefetch_pending, Stats &stats,
                           PreparedEpoch &prepared, TimelineInterval &ready,
                           TimelineInterval &upload,
                           const bool coherent_lookahead,
                           bool &poison) noexcept {
  prepare_detail::Context context{state,
                                  input,
                                  run,
                                  epoch,
                                  prefetch_pending,
                                  stats,
                                  prepared,
                                  ready,
                                  upload,
                                  coherent_lookahead,
                                  poison,
                                  nullptr,
                                  prepare_detail::Backend::Accelerator};
  return prepare_epoch(context);
}

bool wait_prefetch_cpu(residency::Pool &pool, std::array<bool, 2u> &pending,
                       const bool publish, const bool source_known) noexcept {
  return prepare_detail::wait_prefetch(pool, pending, publish, source_known,
                                       prepare_detail::Backend::Cpu);
}

bool wait_prefetch_accel(residency::Pool &pool, std::array<bool, 2u> &pending,
                         const bool publish, const bool source_known) noexcept {
  return prepare_detail::wait_prefetch(pool, pending, publish, source_known,
                                       prepare_detail::Backend::Accelerator);
}

} // namespace rund::compute::detail::virtual_run_overlap
