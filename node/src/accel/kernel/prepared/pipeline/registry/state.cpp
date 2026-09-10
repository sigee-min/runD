#include "../registry.hpp"

#include "../reservation.hpp"
#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <new>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::mul;

PreparedKernelTemplateRegistryState *
registry_state(const PreparedKernelTemplateRegistry &registry) noexcept {
  auto *const state =
      static_cast<PreparedKernelTemplateRegistryState *>(registry.owner.get());
  return state != nullptr && state->magic == kTemplateRegistryMagic ? state
                                                                    : nullptr;
}

bool PreparedKernelTemplateRegistryBytes(const std::uint64_t template_count,
                                         std::uint64_t &bytes) noexcept {
  bytes = sizeof(PreparedKernelTemplateRegistryState);
  std::uint64_t entries = 0u;
  std::uint64_t charges = 0u;
  return mul(template_count, sizeof(PreparedKernelTemplateEntry), entries) &&
         mul(template_count, sizeof(PreparedKernelTemplateCharge), charges) &&
         accumulate(bytes, entries) && accumulate(bytes, charges);
}

rund::AccelCheck BindPreparedKernelTemplateRegistry(
    const rund::AccelApi api, const std::uint64_t context_id,
    PreparedKernelTemplateRegistry &registry) noexcept {
  if ((api != rund::AccelApi::Metal && api != rund::AccelApi::Vulkan) ||
      context_id == 0u) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  if (registry.owner == nullptr) {
    try {
      auto state = std::make_shared<PreparedKernelTemplateRegistryState>();
      state->context_id = context_id;
      state->api = api;
      if (registry.limit.ok) {
        if (registry.limit.template_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        const std::size_t template_count =
            static_cast<std::size_t>(registry.limit.template_count);
        state->entries.reserve(template_count);
        state->template_charges.reserve(template_count);
        // The cumulative transaction starts with the immutable public-plan
        // identity. Charges intentionally carry no fingerprint of their own;
        // accumulation therefore preserves this seed while primary and
        // alternate streams consume the same frozen limit.
        state->consumed.ok = true;
        state->consumed.reason = "ok";
        state->consumed.fingerprint_hi = registry.limit.fingerprint_hi;
        state->consumed.fingerprint_lo = registry.limit.fingerprint_lo;
        state->consumed.template_capacity = registry.limit.template_capacity;
        state->consumed.template_step_capacity =
            registry.limit.template_step_capacity;
        state->consumed.descriptor_set_capacity =
            registry.limit.descriptor_set_capacity;
        state->consumed.descriptor_capacity =
            registry.limit.descriptor_capacity;
      }
      registry.owner = std::move(state);
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || state->context_id != context_id ||
      state->api != api) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  if (registry.limit.ok &&
      (state->consumed.fingerprint_hi != registry.limit.fingerprint_hi ||
       state->consumed.fingerprint_lo != registry.limit.fingerprint_lo ||
       state->consumed.template_capacity != registry.limit.template_capacity ||
       state->consumed.template_step_capacity !=
           registry.limit.template_step_capacity ||
       state->consumed.descriptor_set_capacity !=
           registry.limit.descriptor_set_capacity ||
       state->consumed.descriptor_capacity !=
           registry.limit.descriptor_capacity)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  registry.reservation = state->consumed;
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
