#pragma once

#include "../emit.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::node::accel::detail::recurrence_source_detail {

[[nodiscard]] bool FindOne(std::string_view source, std::string_view needle,
                           std::size_t &at) noexcept;
[[nodiscard]] bool FindInputLoad(std::string_view source, ComputeApi api,
                                 ComputeScalar scalar, std::string_view name,
                                 bool uniform, std::size_t &begin,
                                 std::size_t &end) noexcept;
[[nodiscard]] bool FindOutputStore(std::string_view source, ComputeApi api,
                                   ComputeScalar scalar, std::string_view name,
                                   std::size_t &begin, std::size_t &value_begin,
                                   std::size_t &value_end,
                                   std::size_t &end) noexcept;
[[nodiscard]] bool FindMetalKernelName(std::string_view source,
                                       const RecurrenceSourceRecipe &recipe,
                                       std::size_t &begin,
                                       std::size_t &end) noexcept;
[[nodiscard]] bool
SourceBindings(const rund::kernel::ExecutionMetadata &, std::string_view source,
               ComputeApi api, ComputeScalar scalar,
               std::span<const std::uint64_t> history_pitch_bytes,
               std::array<SourceBinding, RecurrenceBindingCapacity> &inputs,
               std::size_t &input_count,
               std::array<OutputBinding, RecurrenceBindingCapacity> &outputs,
               std::size_t &output_count) noexcept;
[[nodiscard]] bool PopulateRecipeEvents(RecurrenceSourceRecipe &) noexcept;

} // namespace rund::node::accel::detail::recurrence_source_detail
