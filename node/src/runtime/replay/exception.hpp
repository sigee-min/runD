#pragma once

#include <rund/replay/code.hpp>

#include <exception>
#include <new>
#include <stdexcept>

namespace rund::node::replay_detail {

// The result boundary owns semantic Code selection; this value owns the one
// C++ exception-type projection shared by Replay noexcept entrypoints.
struct ExceptionCodes final {
  ::rund::replay::Code bad_alloc{};
  ::rund::replay::Code length_error{};
  ::rund::replay::Code unexpected{};
};

[[nodiscard]] inline ::rund::replay::Code
CurrentExceptionCode(const ExceptionCodes codes) noexcept {
  const std::exception_ptr error = std::current_exception();
  if (error == nullptr) {
    return codes.unexpected;
  }
  try {
    std::rethrow_exception(error);
  } catch (const std::bad_alloc &) {
    return codes.bad_alloc;
  } catch (const std::length_error &) {
    return codes.length_error;
  } catch (...) {
    return codes.unexpected;
  }
}

} // namespace rund::node::replay_detail
