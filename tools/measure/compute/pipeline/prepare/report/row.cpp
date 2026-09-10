#include "row.hpp"

#include "backend.hpp"
#include "failure.hpp"
#include "format.hpp"
#include "memory.hpp"
#include "plan.hpp"

#include <cstdio>

namespace rund::measure::compute::preparation_memory::report {

void PrintObservation(const Backend backend, const bool materialize,
                      const PreparationMemoryObservation &observed) {
  std::printf("prepare_memory,%s,%s,%s", Name(backend),
              materialize ? "materialize" : "plan_only", observed.status);
  PrintPlanFields(observed);
  PrintMemoryFields(observed);
  PrintBackendFields(observed);
  PrintFailureFields(observed);
  std::putchar('\n');
}

} // namespace rund::measure::compute::preparation_memory::report
