#pragma once

// Included by backend/run.hpp inside rund::node::accel::detail.

// Rebuild one already-planned step against an alternate resident binding set.
// Backend View lowering uses this to substitute cold-prepared dense storage
// without creating a second graph, plan, or schedule authority.
[[nodiscard]] bool RebindBoundStep(const BoundStep &source,
                                   const RunBinds &binds, BoundStep &out);

struct BoundRun final {
  BackendRun run{};
  BoundStepStorage storage{};
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";

  BoundRun() = default;
  BoundRun(const BoundRun &) = delete;
  BoundRun &operator=(const BoundRun &) = delete;
  BoundRun(BoundRun &&other) noexcept;
  BoundRun &operator=(BoundRun &&other) noexcept;

  void bind(const KernelExecution &execution,
            std::uint64_t original_dispatch_count,
            std::uint64_t final_dispatch_count) noexcept;
};

[[nodiscard]] BoundRun BuildBoundRun(
    const rund::AccelContext &context, const KernelExecution &execution,
    const RunBinds &binds, const BoundResets &resets,
    const PlannedStepStorage &planned, const ScheduledStepOrder &order,
    std::uint64_t original_dispatch_count, std::uint64_t final_dispatch_count);

[[nodiscard]] rund::kernel::BindingSet
MapBindingFor(const BoundStep &step) noexcept;

template <typename Bindings>
[[nodiscard]] const Bindings *BindingsFor(const BoundStep &step) noexcept {
  return std::get_if<Bindings>(&step.bindings);
}

template <typename Bindings>
[[nodiscard]] const Bindings *
BindingsFor(const BoundStep &step, const rund::kernel::NodeKind kind) noexcept {
  return BoundStepMatches(step, kind) ? BindingsFor<Bindings>(step) : nullptr;
}
