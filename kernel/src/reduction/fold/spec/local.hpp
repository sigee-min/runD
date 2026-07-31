#pragma once

#include "../local.hpp"

namespace rund::kernel::reduction::fold::spec_detail {

FoldPaddingLaw PaddingLaw(FoldOperation operation);
FoldOverflowLaw OverflowLaw(FoldOperation operation);

} // namespace rund::kernel::reduction::fold::spec_detail
