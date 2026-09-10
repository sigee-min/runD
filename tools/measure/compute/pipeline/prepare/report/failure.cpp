#include "failure.hpp"

#include "format.hpp"

#include <string_view>

namespace rund::measure::compute::preparation_memory::report {

void PrintFailureFields(const PreparationMemoryObservation &observed) {
  const auto &location = observed.location;
  PrintUnsigned(static_cast<std::uint64_t>(observed.code));
  PrintUnsigned(static_cast<std::uint64_t>(observed.reason));
  PrintCsvField(observed.error);
  PrintUnsigned(location.known() ? 1u : 0u);
  PrintUnsigned(location.step);
  PrintUnsigned(location.iteration);
  PrintUnsigned(location.node);
  PrintUnsigned(location.template_index);
  PrintUnsigned(location.occurrence_index);
  PrintUnsigned(location.outer_iteration);
  PrintUnsigned(location.inner_iteration);
  PrintUnsigned(static_cast<std::uint64_t>(location.nested_phase));
  PrintCsvField(location.native_reason_key == nullptr
                    ? std::string_view{}
                    : std::string_view{location.native_reason_key});
}

} // namespace rund::measure::compute::preparation_memory::report
