#pragma once

#include <exception>
#include <new>
#include <stdexcept>

namespace rund::node::accel::detail::backend_exception {

// Capacity-producing C++ exception classes have one backend-neutral owner.
// Callers retain their own cleanup and result vocabulary. Any other active
// exception is rethrown with its original identity.
inline void RethrowUnlessCapacityException() {
  const std::exception_ptr error = std::current_exception();
  if (error == nullptr) {
    std::terminate();
  }
  try {
    std::rethrow_exception(error);
  } catch (const std::bad_alloc &) {
    return;
  } catch (const std::length_error &) {
    return;
  }
}

} // namespace rund::node::accel::detail::backend_exception
