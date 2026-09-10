#pragma once

// DeviceVSM validation is intentionally split by authority.  Keep this
// aggregate for existing callers; implementations live in the focused source
// owners below.
#include "validation/proof.hpp"
#include "validation/request.hpp"
