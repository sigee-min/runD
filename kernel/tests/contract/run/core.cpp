#include "contract/cases.hpp"

int main() {
  if (const int result = RunCheckedContract(); result != 0) {
    return result;
  }
  return RunOrchestratorContract();
}
