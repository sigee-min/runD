#pragma once

#include "../graph.hpp"
#include "collective.hpp"
#include "map.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail {

struct ResetRoute final {
  std::uint32_t value_index{};
  std::uint32_t step{};
  std::uint32_t last{};
};

// Immutable compiled CPU graph owner. Runtime values and step plans are
// published once here; mutable execution routes belong to CpuGraphStorage.
struct CpuGraphProgram final {
  ~CpuGraphProgram();

  std::unique_ptr<CpuRuntimeGraph> runtime;
  std::vector<std::unique_ptr<CpuProgram>> maps;
  std::vector<std::unique_ptr<CpuCollective>> collectives;
  std::vector<std::size_t> bind_begin;
  std::vector<std::size_t> bind_count;
  std::vector<ResetRoute> resets;
  std::uint64_t graph_hash{};
};

} // namespace rund::compute::detail
