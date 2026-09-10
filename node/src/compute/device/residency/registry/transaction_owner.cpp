#include "transaction_owner.hpp"

namespace rund::compute::detail::residency {

VirtualTransactionOwner::VirtualTransactionOwner(Authority &authority) noexcept
    : authority_(authority) {}

VirtualTransactionOwner Authority::virtual_transactions() noexcept {
  return VirtualTransactionOwner{*this};
}

} // namespace rund::compute::detail::residency
