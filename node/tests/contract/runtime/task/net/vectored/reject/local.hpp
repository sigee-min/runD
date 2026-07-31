#pragma once

#include "../local.hpp"

[[nodiscard]] bool NetVectoredRejectsEmptySlices();
[[nodiscard]] bool NetVectoredRejectsNullSlices();
[[nodiscard]] bool NetVectoredRejectsCapacityOverflow();
[[nodiscard]] bool NetVectoredReportsWouldBlock();
