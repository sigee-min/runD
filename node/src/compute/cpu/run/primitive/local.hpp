#pragma once

#include "model.hpp"

#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::compute::detail {

[[nodiscard]] Status bind_cpu_primitive_ports(JobState &, CpuGraphProgram &,
                                              CpuGraphRun &, std::size_t,
                                              std::span<RawCpuBuffer>) noexcept;

[[nodiscard]] Status read_cpu_primitive_count(const PrimitiveContext &,
                                              std::uint32_t value,
                                              std::size_t capacity,
                                              std::uint32_t &count) noexcept;

[[nodiscard]] Status finish_cpu_primitive(bool ok,
                                          std::string_view reason) noexcept;

[[nodiscard]] Status run_cpu_primitive_collective(PrimitiveContext &);
[[nodiscard]] Status run_cpu_primitive_reference(PrimitiveContext &);
[[nodiscard]] Status run_cpu_primitive_indexed(PrimitiveContext &);
[[nodiscard]] Status run_cpu_primitive_ordering(PrimitiveContext &);

} // namespace rund::compute::detail
