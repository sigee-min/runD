#include "residency.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] std::size_t
find_frame(const std::vector<Authority::Frame> &frames,
           const CacheKey key) noexcept {
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    if (frames[index].state != FrameState::Empty && frames[index].key == key) {
      return index;
    }
  }
  return frames.size();
}

[[nodiscard]] bool demanded(const std::span<const CacheUse> uses,
                            const CacheKey key) noexcept {
  return std::any_of(uses.begin(), uses.end(),
                     [key](const CacheUse use) { return use.key == key; });
}

[[nodiscard]] std::size_t
select_frame(const std::vector<Authority::Frame> &frames,
             const std::span<const CacheUse> uses) noexcept {
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    if (frames[index].state == FrameState::Empty) {
      return index;
    }
  }
  std::size_t selected = frames.size();
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    const Authority::Frame &frame = frames[index];
    if (demanded(uses, frame.key)) {
      continue;
    }
    if (selected == frames.size() ||
        frame.next_use > frames[selected].next_use ||
        (frame.next_use == frames[selected].next_use &&
         frames[selected].key < frame.key)) {
      selected = index;
    }
  }
  return selected;
}

} // namespace

bool Authority::configure(const std::uint64_t page_bytes,
                          const std::uint32_t frame_capacity) noexcept {
  if (page_bytes == 0u || frame_capacity == 0u) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (active_token_ != 0u) {
    return false;
  }
  if (page_bytes_ != 0u) {
    return page_bytes_ == page_bytes && frames_.size() == frame_capacity;
  }
  try {
    frames_.resize(frame_capacity);
    rollback_.resize(frame_capacity);
    bindings_.reserve(frame_capacity);
    transitions_.reserve(static_cast<std::size_t>(frame_capacity) * 4u);
    page_bytes_ = page_bytes;
    return true;
  } catch (const std::bad_alloc &) {
    frames_.clear();
    rollback_.clear();
    bindings_.clear();
    transitions_.clear();
    return false;
  }
}

AuthorityResult
Authority::begin(const std::span<const CacheUse> uses) noexcept {
  std::lock_guard lock{gate_};
  if (active_token_ != 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Busy};
  }
  if (page_bytes_ == 0u || uses.empty() || uses.size() > frames_.size()) {
    return AuthorityResult{.failure = uses.size() > frames_.size()
                                          ? AuthorityFailure::Capacity
                                          : AuthorityFailure::Invalid};
  }
  for (std::size_t left = 0u; left < uses.size(); ++left) {
    if (uses[left].key.backing == 0u) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    for (std::size_t right = left + 1u; right < uses.size(); ++right) {
      if (uses[left].key == uses[right].key) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
    }
  }

  rollback_ = frames_;
  bindings_.clear();
  transitions_.clear();
  for (const CacheUse use : uses) {
    std::size_t frame = find_frame(frames_, use.key);
    bool fetch = false;
    if (frame == frames_.size()) {
      frame = select_frame(frames_, uses);
      if (frame == frames_.size()) {
        frames_ = rollback_;
        return AuthorityResult{.failure = AuthorityFailure::Capacity};
      }
      if (frames_[frame].state != FrameState::Empty) {
        if (frames_[frame].dirty) {
          transitions_.push_back(CacheTransition{
              .key = frames_[frame].key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Writeback,
          });
        }
        transitions_.push_back(CacheTransition{
            .key = frames_[frame].key,
            .frame = static_cast<std::uint32_t>(frame),
            .kind = TransitionKind::Unmap,
        });
      }
      transitions_.push_back(CacheTransition{
          .key = use.key,
          .frame = static_cast<std::uint32_t>(frame),
          .kind = TransitionKind::Fetch,
      });
      transitions_.push_back(CacheTransition{
          .key = use.key,
          .frame = static_cast<std::uint32_t>(frame),
          .kind = TransitionKind::Map,
      });
      frames_[frame] = Frame{
          .key = use.key,
          .next_use = use.next_use,
          .state = FrameState::Mapping,
      };
      fetch = true;
    } else {
      frames_[frame].next_use = use.next_use;
      frames_[frame].state = FrameState::Pinned;
    }
    bindings_.push_back(CacheBinding{
        .key = use.key,
        .frame = static_cast<std::uint32_t>(frame),
        .access = use.access,
        .fetch = fetch,
    });
  }
  active_token_ = next_token_++;
  active_drain_ = false;
  if (next_token_ == 0u) {
    next_token_ = 1u;
  }
  return AuthorityResult{
      .failure = AuthorityFailure::None,
      .lease =
          EpochLease{
              .bindings = bindings_,
              .transitions = transitions_,
              .token = active_token_,
          },
  };
}

AuthorityResult Authority::drain_dirty() noexcept {
  std::lock_guard lock{gate_};
  if (active_token_ != 0u || page_bytes_ == 0u) {
    return AuthorityResult{.failure = active_token_ != 0u
                                          ? AuthorityFailure::Busy
                                          : AuthorityFailure::Invalid};
  }
  rollback_ = frames_;
  bindings_.clear();
  transitions_.clear();
  for (std::size_t index = 0u; index < frames_.size(); ++index) {
    if (!frames_[index].dirty) {
      continue;
    }
    transitions_.push_back(CacheTransition{
        .key = frames_[index].key,
        .frame = static_cast<std::uint32_t>(index),
        .kind = TransitionKind::Writeback,
    });
  }
  active_token_ = next_token_++;
  active_drain_ = true;
  if (next_token_ == 0u) {
    next_token_ = 1u;
  }
  return AuthorityResult{
      .failure = AuthorityFailure::None,
      .lease = EpochLease{.bindings = bindings_,
                          .transitions = transitions_,
                          .token = active_token_},
  };
}

bool Authority::complete(const std::uint64_t token,
                         const bool success) noexcept {
  std::lock_guard lock{gate_};
  if (token == 0u || token != active_token_) {
    return false;
  }
  if (!success) {
    frames_ = rollback_;
    if (!active_drain_) {
      for (const CacheBinding binding : bindings_) {
        if (binding.fetch) {
          frames_[binding.frame] = Frame{};
        }
      }
    }
  } else if (active_drain_) {
    for (const CacheTransition transition : transitions_) {
      Frame &frame = frames_[transition.frame];
      if (transition.kind == TransitionKind::Writeback &&
          frame.key == transition.key) {
        frame.dirty = false;
        frame.state = FrameState::DeviceReady;
      }
    }
  } else {
    for (const CacheBinding binding : bindings_) {
      Frame &frame = frames_[binding.frame];
      frame.state =
          writes(binding.access) ? FrameState::Dirty : FrameState::DeviceReady;
      frame.dirty = frame.dirty || writes(binding.access);
    }
  }
  active_token_ = 0u;
  active_drain_ = false;
  return true;
}

bool Authority::discard(const std::uint64_t token) noexcept {
  std::lock_guard lock{gate_};
  if (token == 0u || token != active_token_) {
    return false;
  }
  frames_ = rollback_;
  for (const CacheTransition transition : transitions_) {
    if (transition.frame < frames_.size() &&
        transition.kind == TransitionKind::Writeback &&
        frames_[transition.frame].key == transition.key) {
      frames_[transition.frame] = Frame{};
    }
  }
  for (const CacheBinding binding : bindings_) {
    if (binding.fetch && binding.frame < frames_.size()) {
      frames_[binding.frame] = Frame{};
    }
  }
  active_token_ = 0u;
  active_drain_ = false;
  return true;
}

bool Authority::probe(const std::span<const CacheKey> keys,
                      const std::span<std::uint8_t> resident) const noexcept {
  if (keys.size() != resident.size()) {
    return false;
  }
  std::lock_guard lock{gate_};
  for (std::size_t index = 0u; index < keys.size(); ++index) {
    resident[index] = static_cast<std::uint8_t>(
        find_frame(frames_, keys[index]) != frames_.size());
  }
  return true;
}

std::uint64_t Authority::page_bytes() const noexcept {
  std::lock_guard lock{gate_};
  return page_bytes_;
}

std::uint32_t Authority::frame_capacity() const noexcept {
  std::lock_guard lock{gate_};
  return static_cast<std::uint32_t>(frames_.size());
}

} // namespace rund::compute::detail::residency
