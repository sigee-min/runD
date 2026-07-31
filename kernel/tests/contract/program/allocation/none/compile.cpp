#include "compile/local.hpp"

int RunProgramNoAllocationCompileRunContract() {
  if (program_no_allocation_contract::Policy() != 0) {
    return 1;
  }
  if (program_no_allocation_contract::Failure() != 0) {
    return 1;
  }
  return 0;
}
