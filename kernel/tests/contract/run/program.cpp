#include "contract/program/cases.hpp"

int main() {
  if (const int rc = RunProgramNoAllocationContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunProgramSkeletonContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunProgramStrictFloatContract(); rc != 0) {
    return rc;
  }
  return 0;
}
