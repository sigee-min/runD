#pragma once

#include <rund/compute/backend.hpp>

namespace rund::node::test_contract::numeric {

[[nodiscard]] int CheckLane32();
[[nodiscard]] int CheckLane64();
[[nodiscard]] int CheckFormats();

[[nodiscard]] int CheckModesLane32();
[[nodiscard]] int CheckModesLane64();
[[nodiscard]] int CheckModesFormat32();
[[nodiscard]] int CheckModesFormat64();
[[nodiscard]] bool CheckTransformRejections();
[[nodiscard]] int CheckWindow32();
[[nodiscard]] int CheckWindow64();
[[nodiscard]] int CheckTransform32();
[[nodiscard]] int CheckTransform64();
[[nodiscard]] int CheckAdditional32();
[[nodiscard]] int CheckAdditional64();
[[nodiscard]] int CheckGraph32();
[[nodiscard]] int CheckGraph64();

[[nodiscard]] int CheckGolden32(rund::compute::Backend backend);
[[nodiscard]] int CheckGolden64(rund::compute::Backend backend);
[[nodiscard]] int CheckMatrix32(rund::compute::Backend backend);
[[nodiscard]] int CheckMatrix64(rund::compute::Backend backend);
[[nodiscard]] int CheckSolve32(rund::compute::Backend backend);
[[nodiscard]] int CheckSolve64(rund::compute::Backend backend);
[[nodiscard]] int CheckStored32(rund::compute::Backend backend);
[[nodiscard]] int CheckStored64(rund::compute::Backend backend);

[[nodiscard]] int RunAccel();
[[nodiscard]] int RunModes();

} // namespace rund::node::test_contract::numeric
