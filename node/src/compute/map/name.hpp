#pragma once

#include <rund/compute/abi/model.hpp>

#include <array>
#include <cstddef>

namespace rund::compute::detail {

inline constexpr std::size_t MapInputCapacity = MaxMapInputs * 2u;

static_assert(MapInputCapacity == 32u);

[[nodiscard]] inline const char *
map_input_name(const std::size_t count, const std::size_t index) noexcept {
  static constexpr std::array<const char *, MapInputCapacity> names{
      "input0",  "input1",  "input2",  "input3",  "input4",  "input5",
      "input6",  "input7",  "input8",  "input9",  "input10", "input11",
      "input12", "input13", "input14", "input15", "input16", "input17",
      "input18", "input19", "input20", "input21", "input22", "input23",
      "input24", "input25", "input26", "input27", "input28", "input29",
      "input30", "input31"};
  return count == 1u ? "input" : index < names.size() ? names[index] : nullptr;
}

[[nodiscard]] inline const char *
map_output_name(const std::size_t count, const std::size_t index) noexcept {
  static constexpr std::array<const char *, MaxOutputs> names{
      "output0",  "output1",  "output2",  "output3", "output4",  "output5",
      "output6",  "output7",  "output8",  "output9", "output10", "output11",
      "output12", "output13", "output14", "output15"};
  return count == 1u ? "output" : index < names.size() ? names[index] : nullptr;
}

} // namespace rund::compute::detail
