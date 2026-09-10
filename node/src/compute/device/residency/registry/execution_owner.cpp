#include "execution_owner.hpp"

namespace rund::compute::detail::residency {

ExecutionOwner::ExecutionOwner(Authority &authority) noexcept
    : authority_(authority) {}

ExecutionOwner Authority::executions() noexcept {
  return ExecutionOwner{*this};
}

} // namespace rund::compute::detail::residency
