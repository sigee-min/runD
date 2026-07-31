#pragma once

#include <kernel/reduction/fold/strict.hpp>

namespace rund::kernel::strict_float {

u64 AddFloat32Bits(u64 left_bits, u64 right_bits,
                   StrictFloatReductionPolicy policy);
u64 AddFloat64Bits(u64 left_bits, u64 right_bits,
                   StrictFloatReductionPolicy policy);

} // namespace rund::kernel::strict_float
