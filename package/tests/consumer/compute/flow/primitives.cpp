#include "primitives/model.hpp"

namespace package_compute {

int FlowPrimitives() {
  if (const int result = Filter(); result != 0)
    return result;
  if (const int result = Record(); result != 0)
    return result;
  if (const int result = Group(); result != 0)
    return result;
  if (const int result = Collective(); result != 0)
    return result;
  if (const int result = Compact(); result != 0)
    return result;
  if (const int result = Basic(); result != 0)
    return result;
  if (const int result = Math(); result != 0)
    return result;
  return Program();
}

} // namespace package_compute
