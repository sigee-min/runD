#include <rund/reason.hpp>

namespace rund {
namespace {

struct ReasonInfo {
  const char *text = nullptr;
  bool prepared_memory = false;
};

ReasonInfo FindReason(const ReasonCode code) noexcept {
  switch (code) {
#define RUND_NODE_REASON_CATEGORY_Universal true
#define RUND_NODE_REASON_CATEGORY_General false
#define RUND_NODE_REASON_CATEGORY_PreparedMemory true
#define RUND_NODE_REASON(value, name, text, category)                          \
  case ReasonCode::name:                                                       \
    return ReasonInfo{text, RUND_NODE_REASON_CATEGORY_##category};
#include <rund/reason.def>
#undef RUND_NODE_REASON
#undef RUND_NODE_REASON_CATEGORY_PreparedMemory
#undef RUND_NODE_REASON_CATEGORY_General
#undef RUND_NODE_REASON_CATEGORY_Universal
  }
  return {};
}

} // namespace

const char *ReasonString(const ReasonCode code) noexcept {
  const ReasonInfo info = FindReason(code);
  return info.text == nullptr ? "unknown_reason_code" : info.text;
}

bool ValidReasonCode(const ReasonCode code) noexcept {
  return FindReason(code).text != nullptr;
}

bool ValidPreparedMemoryReason(const ReasonCode code) noexcept {
  const ReasonInfo info = FindReason(code);
  return info.text != nullptr && info.prepared_memory;
}

} // namespace rund
