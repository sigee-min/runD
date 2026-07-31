#pragma once

#include <rund/task/handle.hpp>
#include <rund/task/handle/ref.hpp>

namespace rund::detail::task {

struct Spawned final {
  ::rund::task::Handle task{};
  ResultRef result{};
};

} // namespace rund::detail::task
