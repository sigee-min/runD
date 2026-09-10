#include "internal.hpp"

#include <algorithm>
#include <array>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
fail_service(PersistentResidencySlidingControl &control,
             const PersistentResidencySlidingServiceFailure &failure) noexcept {
  if (failure.check.ok) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  const Binding binding = bind(control);
  const std::shared_ptr<Owner> owner = binding.owner;
  if (owner == nullptr || owner->adapter == nullptr) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  PersistentResidencySlidingFinal final{};
  {
    std::unique_lock terminal_gate{owner->adapter->residency_terminal_gate};
    std::unique_lock lock{owner->gate};
    bool closed = false;
    {
      std::lock_guard control_lock{control.gate};
      closed = !control.active || control.final_callback_count != 0u ||
               control.quarantined;
    }
    if (!valid_binding(binding, *owner, control) || closed ||
        !persistent_sliding_accept_service_failure(control, failure)) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    {
      std::lock_guard control_lock{control.gate};
      if (control.first_failure.ok) {
        control.first_failure = failure.check;
        control.first_failure_coordinate = failure.identity.coordinate;
      }
    }
    if (failure.terminal == NativeTerminal::Known) {
      return {true, "ok"};
    }
    // A failed Host service cannot safely author Q-W independent descriptors
    // into the W reusable gate rows. Leave every guard closed and advance only
    // the already-encoded event dependencies. The retained command may still
    // touch reset state, so the only honest terminal is UnknownMayWrite.
    std::array<std::uint64_t, PersistentResidencySlidingCapacity> last_ready{};
    const std::size_t count = std::min<std::size_t>(
        SlotCount, static_cast<std::size_t>(owner->prepared.coordinate_count));
    for (std::size_t index = 0u; index < count; ++index) {
      const Coordinate &entry = owner->coordinates[index];
      last_ready[entry.identity.slot] = entry.ready_value;
      if (entry.sequence != nullptr && entry.sequence->guard_zero != nil &&
          [entry.sequence->guard_zero contents] != nullptr) {
        *static_cast<std::uint32_t *>([entry.sequence->guard_zero contents]) =
            1u;
      }
    }
    {
      std::lock_guard control_lock{control.gate};
      control.final_callback_count = 1u;
    }
    quarantine_unknown(*owner, control);
    for (std::size_t slot = 0u; slot < owner->prepared.width; ++slot) {
      owner->ready[slot].signaledValue = last_ready[slot];
    }
    final = make_final(*owner, control, failure.check,
                       NativeTerminal::UnknownMayWrite,
                       failure.identity.coordinate);
  }
  deliver_final(*owner, std::move(final));
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
