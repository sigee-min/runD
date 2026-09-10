#include "local.hpp"

#include "../../../suite/core.hpp"

#include <chrono>
#include <span>
#include <utility>
#include <vector>

namespace rund::measure::compute::virtual_graph_residency::run_detail {

[[nodiscard]] double micros(const Clock::duration duration) noexcept {
  return std::chrono::duration<double, std::micro>(duration).count();
}

[[nodiscard]] std::uint64_t nanos() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

[[nodiscard]] double phase_us(const std::uint64_t begin,
                              const std::uint64_t end) noexcept {
  return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
}

[[nodiscard]] bool same_owner(const std::shared_ptr<Owner> &owner,
                              const std::uintptr_t owner_id,
                              const std::uintptr_t control_id) noexcept {
  return owner != nullptr &&
         reinterpret_cast<std::uintptr_t>(owner.get()) == owner_id &&
         reinterpret_cast<std::uintptr_t>(owner->registration.get()) ==
             control_id;
}

[[nodiscard]] bool phase_ready(
    const ::rund::compute::Status status, const std::shared_ptr<Owner> &owner,
    const ::rund::node::accel::detail::DeviceVsmEvidence &native,
    const std::uint64_t enter, const std::uint64_t run_end,
    const std::uintptr_t owner_id, const std::uintptr_t control_id) noexcept {
  return static_cast<bool>(status) && same_owner(owner, owner_id, control_id) &&
         owner->evidence != nullptr && owner->evidence->final_received &&
         native.native_check_ok && native.native_submit_count == 1u &&
         native.final_callback_count == 1u && enter != 0u &&
         native.submit_ns != 0u && native.callback_ns != 0u &&
         native.final_ns != 0u && run_end != 0u && enter <= native.submit_ns &&
         native.submit_ns <= native.callback_ns &&
         native.callback_ns <= native.final_ns && native.final_ns <= run_end;
}

} // namespace rund::measure::compute::virtual_graph_residency::run_detail
