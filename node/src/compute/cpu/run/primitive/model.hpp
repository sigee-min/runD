#pragma once

#include <cstddef>
#include <span>

namespace rund::compute::detail {

struct CpuGraphProgram;
struct CpuGraphRun;
struct CpuRuntimePrimitive;
struct JobState;

struct RawCpuBuffer final {
  std::byte *data = nullptr;
  std::size_t bytes = 0u;

  [[nodiscard]] explicit operator bool() const noexcept {
    return data != nullptr;
  }
};

struct PrimitiveContext final {
  JobState &job;
  CpuGraphProgram &cpu;
  CpuGraphRun &run;
  std::size_t step = 0u;
  const CpuRuntimePrimitive &primitive;
  std::span<const RawCpuBuffer> ports{};

  [[nodiscard]] const RawCpuBuffer &port(std::size_t index) const noexcept {
    return ports[index];
  }
};

} // namespace rund::compute::detail
