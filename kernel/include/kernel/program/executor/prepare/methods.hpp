#pragma once

#include <kernel/program/executor/prepare/narrow.hpp>

namespace rund::kernel {

template <std::size_t Rank>
[[nodiscard]] inline PreparedEach<Rank> Executor::prepare(
    const Space<Rank>& index_space) const {
  return prepare_each(*this, index_space);
}

} // namespace rund::kernel
