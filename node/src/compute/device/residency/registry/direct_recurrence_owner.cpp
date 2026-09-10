#include "direct_recurrence_owner.hpp"

namespace rund::compute::detail::residency {

DirectRecurrenceOwner::DirectRecurrenceOwner(Authority &authority) noexcept
    : authority_(authority) {}

DirectRecurrenceOwner Authority::direct_recurrences() noexcept {
  return DirectRecurrenceOwner{*this};
}

} // namespace rund::compute::detail::residency
