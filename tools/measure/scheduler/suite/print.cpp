#include "model.hpp"

namespace rund::measure::scheduler {

void Print(const char *const name, const Measure &value) {
  const double cold_per_op = value.ops == 0u ? 0.0 : value.cold_ns / value.ops;
  const double warm_per_op = value.ops == 0u ? 0.0 : value.warm_ns / value.ops;
  std::printf(
      "%s ok=%u reason=%.*s ops=%llu cold_ns_per_op=%.3f "
      "warm_median_ns_per_op=%.3f spawned=%llu completed=%llu parked=%llu "
      "resumed=%llu workers=%llu participating=%llu batches=%llu "
      "batch_tasks=%llu frames_alloc=%llu frames_reuse=%llu\n",
      name, value.ok ? 1u : 0u, static_cast<int>(value.reason.size()),
      value.reason.data(), static_cast<unsigned long long>(value.ops),
      cold_per_op, warm_per_op,
      static_cast<unsigned long long>(value.stats.spawned()),
      static_cast<unsigned long long>(value.stats.completed()),
      static_cast<unsigned long long>(value.stats.parked()),
      static_cast<unsigned long long>(value.stats.resumed()),
      static_cast<unsigned long long>(value.stats.task_workers()),
      static_cast<unsigned long long>(value.stats.participating_task_workers()),
      static_cast<unsigned long long>(
          value.stats.lane_dispatch_batch_packets()),
      static_cast<unsigned long long>(
          value.stats.lane_dispatch_batch_logical_tasks()),
      static_cast<unsigned long long>(
          value.stats.resources().coroutine_frame_allocations()),
      static_cast<unsigned long long>(
          value.stats.resources().coroutine_frame_reuses()));
}

void PrintScale(const ScaleMeasure &value) {
  const Measure &total = value.total;
  const double divisor = total.ops == 0u ? 1.0 : total.ops;
  std::printf(
      "task_scale ok=%u reason=%.*s ops=%llu payload_ops=%u "
      "cold_ns_per_op=%.3f warm_median_ns_per_op=%.3f "
      "cold_admit_ns_per_op=%.3f warm_admit_ns_per_op=%.3f "
      "cold_drain_ns_per_op=%.3f warm_drain_ns_per_op=%.3f "
      "spawned=%llu completed=%llu parked=%llu resumed=%llu workers=%llu "
      "participating=%llu batches=%llu batch_tasks=%llu frames_alloc=%llu "
      "frames_reuse=%llu\n",
      total.ok ? 1u : 0u, static_cast<int>(total.reason.size()),
      total.reason.data(), static_cast<unsigned long long>(total.ops),
      value.payload_ops, total.cold_ns / divisor, total.warm_ns / divisor,
      value.cold_admit_ns / divisor, value.warm_admit_ns / divisor,
      value.cold_drain_ns / divisor, value.warm_drain_ns / divisor,
      static_cast<unsigned long long>(total.stats.spawned()),
      static_cast<unsigned long long>(total.stats.completed()),
      static_cast<unsigned long long>(total.stats.parked()),
      static_cast<unsigned long long>(total.stats.resumed()),
      static_cast<unsigned long long>(total.stats.task_workers()),
      static_cast<unsigned long long>(total.stats.participating_task_workers()),
      static_cast<unsigned long long>(
          total.stats.lane_dispatch_batch_packets()),
      static_cast<unsigned long long>(
          total.stats.lane_dispatch_batch_logical_tasks()),
      static_cast<unsigned long long>(
          total.stats.resources().coroutine_frame_allocations()),
      static_cast<unsigned long long>(
          total.stats.resources().coroutine_frame_reuses()));
}


} // namespace rund::measure::scheduler
