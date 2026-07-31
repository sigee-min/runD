#pragma once

#include "contract/program/compute/backend/lowering/local.hpp"
#include "contract/program/compute/reject/model.hpp"

namespace program_compute_contract::lowering_reject {

int Binding();
int Carrier();
int Domain();
int Shape();
int Storage();

} // namespace program_compute_contract::lowering_reject
