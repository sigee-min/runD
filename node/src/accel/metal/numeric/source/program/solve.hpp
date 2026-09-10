#pragma once

#include "solve/direct.hpp"
#include "solve/factor.hpp"
#include "solve/kernel.hpp"

#include <string_view>

namespace rund::node::accel::detail::source::program {

template <typename Sink>
[[nodiscard]] inline bool AppendSolveSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(solve::Factor) && sink.append(solve::Direct) &&
         sink.append(solve::Kernel);
}

} // namespace rund::node::accel::detail::source::program
