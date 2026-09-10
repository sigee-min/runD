#include "local.hpp"

namespace rund_node_memory_contract {

int CheckCpuPrimitiveScratchOwnership() {
  if (const int range = cpu_primitive::CheckRange(); range != 0) {
    return 80 + range;
  }
  if (const int empty = cpu_primitive::CheckEmpty(); empty != 0) {
    return empty;
  }
  if (const int scatter = cpu_primitive::CheckScatter(); scatter != 0) {
    return scatter;
  }
  if (const int lane32 = cpu_primitive::CheckTypedI32(); lane32 != 0) {
    return 10 + lane32;
  }
  if (const int lane64 = cpu_primitive::CheckTypedI64(); lane64 != 0) {
    return 30 + lane64;
  }
  return 0;
}

} // namespace rund_node_memory_contract
