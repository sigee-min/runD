#include "blackbox/model.hpp"

using namespace package_blackbox;

int main() {
  if (const int result = CheckRunReplay(); result != 0) {
    return result;
  }
  if (const int result = CheckMathAndEvidence(); result != 0) {
    return result;
  }
  if (const int result = CheckCluster(); result != 0) {
    return result;
  }
  if (const int result = CheckNetwork(); result != 0) {
    return result;
  }
  return 0;
}
