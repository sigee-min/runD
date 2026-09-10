#pragma once

// The fixed-point runner is macro-specialized by the 32/64-bit translation
// units. Keep arithmetic definitions before the executor table in body.hpp.
#include "fixed/base.hpp"
#include "fixed/arithmetic.hpp"
#include "fixed/transcendental.hpp"
