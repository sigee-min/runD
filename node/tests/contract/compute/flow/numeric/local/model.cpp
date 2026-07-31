#include "model.hpp"

#include "../../../../target/selection.hpp"

#include <array>
#include <cstdio>

namespace rund::node::test_contract::numeric {

int RunAccel() {
  if (const int narrow = CheckLane32(); narrow != 0) {
    std::fprintf(stderr, "typed numeric lane32=%d\n", narrow);
    return narrow;
  }
  if (const int wide = CheckLane64(); wide != 0) {
    std::fprintf(stderr, "typed numeric lane64=%d\n", wide);
    return 50 + wide;
  }
  if (const int formats = CheckFormats(); formats != 0) {
    std::fprintf(stderr, "typed numeric arbitrary-format=%d\n", formats);
    return 1000 + formats;
  }
  return 0;
}

int RunModes() {
  using rund::compute::Backend;
  if (const int result = CheckModesLane32(); result != 0) {
    std::fprintf(stderr, "numeric modes fixed_lane32=%d\n", result);
    return result;
  }
  if (const int result = CheckModesLane64(); result != 0) {
    std::fprintf(stderr, "numeric modes fixed_lane64=%d\n", result);
    return 20 + result;
  }
  if (const int result = CheckModesFormat32(); result != 0) {
    std::fprintf(stderr, "numeric modes fixed_16_16=%d\n", result);
    return 40 + result;
  }
  if (const int result = CheckModesFormat64(); result != 0) {
    std::fprintf(stderr, "numeric modes fixed_20_44=%d\n", result);
    return 60 + result;
  }
  if (!CheckTransformRejections()) {
    return 80;
  }
  if (const int result = CheckWindow32(); result != 0) {
    return 90 + result;
  }
  if (const int result = CheckWindow64(); result != 0) {
    return 100 + result;
  }
  if (const int result = CheckTransform32(); result != 0) {
    return 110 + result;
  }
  if (const int result = CheckTransform64(); result != 0) {
    return 120 + result;
  }
  if (const int result = CheckAdditional32(); result != 0) {
    return 130 + result;
  }
  if (const int result = CheckAdditional64(); result != 0) {
    return 140 + result;
  }
  if (const int result = CheckGraph32(); result != 0) {
    return 150 + result;
  }
  if (const int result = CheckGraph64(); result != 0) {
    return 170 + result;
  }

  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (const int result = CheckGolden32(backend); result != 0) {
      std::fprintf(stderr, "numeric golden backend=%u fixed_16_16=%d\n",
                   static_cast<unsigned>(backend), result);
      return 200 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckGolden64(backend); result != 0) {
      std::fprintf(stderr, "numeric golden backend=%u fixed_20_44=%d\n",
                   static_cast<unsigned>(backend), result);
      return 250 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckMatrix32(backend); result != 0) {
      return 300 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckMatrix64(backend); result != 0) {
      return 350 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckSolve32(backend); result != 0) {
      return 400 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckSolve64(backend); result != 0) {
      return 450 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckStored32(backend); result != 0) {
      return 500 + 10 * static_cast<int>(backend) + result;
    }
    if (const int result = CheckStored64(backend); result != 0) {
      return 550 + 10 * static_cast<int>(backend) + result;
    }
  }
  return 0;
}

} // namespace rund::node::test_contract::numeric
