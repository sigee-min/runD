#pragma once

#include <exception>
#include <new>
#include <stdexcept>

namespace rund::compute::detail::compute_exception {

// Compute boundaries opt into this classifier only when both standard
// allocation and container-length failures have the same capacity meaning.
// The caller still owns rollback and Reason selection. Every other active
// exception is rethrown with its original type and payload.
inline void rethrow_unless_capacity_exception() {
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

} // namespace rund::compute::detail::compute_exception
