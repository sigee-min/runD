#pragma once

#include "../local.hpp"

#include "src/compute/program/state.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace rund_node_test_pipeline::memory {

[[nodiscard]] std::shared_ptr<rund::compute::detail::ProgramState>
MakeProgram(rund::compute::Device &,
            std::span<const std::size_t> chunk_counts);

[[nodiscard]] int CheckAttribution();
[[nodiscard]] int CheckSharedAttribution();
[[nodiscard]] int CheckBasic(rund::compute::Device &);
[[nodiscard]] int CheckReset(rund::compute::Device &);
[[nodiscard]] int CheckArena(rund::compute::Device &);
[[nodiscard]] int CheckCapacity(rund::compute::Device &);

} // namespace rund_node_test_pipeline::memory
