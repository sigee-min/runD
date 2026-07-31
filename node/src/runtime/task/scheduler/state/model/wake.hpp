#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::node {

struct TaskRecord;

struct ExternalWake final {
  TaskRecord *record = nullptr;
  std::uint64_t id = 0u;
  std::size_t lane = 0u;
};

} // namespace rund::node
