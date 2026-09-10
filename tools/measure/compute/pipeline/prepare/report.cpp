#include "report.hpp"

#include "report/columns.hpp"
#include "report/row.hpp"

namespace rund::measure::compute::preparation_memory {

void PrintObservation(const Backend backend, const bool materialize,
                      const PreparationMemoryObservation &observed) {
  report::PrintObservation(backend, materialize, observed);
}

} // namespace rund::measure::compute::preparation_memory

namespace rund::measure::compute {

void PrintPreparationMemoryColumns() {
  preparation_memory::report::PrintColumns();
}

} // namespace rund::measure::compute
