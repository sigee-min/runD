#include "../../registry.hpp"

#include "../frame.hpp"
#include "../internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::compute::detail::residency {

void clear(Authority::LeaseSlot &slot) noexcept {
  slot.undo_frames.clear();
  slot.undo.clear();
  slot.bindings.clear();
  slot.transitions.clear();
  slot.ports.clear();
  slot.remaps.clear();
  slot.relocations.clear();
  slot.relocation_frames.clear();
  slot.token = 0u;
  slot.generation = 0u;
  slot.coordinate = 0u;
  slot.book_domain = 0u;
  slot.retry_region = {};
  slot.cpu_key = {};
  slot.plan = {};
  slot.identity = {};
  slot.cycle = 0u;
  slot.state = Authority::LeaseState::Free;
  slot.cpu_bound = false;
  slot.terminal = false;
}

void save(Authority::LeaseSlot &slot, const std::uint32_t frame,
          const Authority::Frame &prior) {
  slot.undo_frames.push_back(frame);
  slot.undo.push_back(prior);
}

void rollback(std::vector<Authority::Frame> &frames, Authority::LeaseSlot &slot,
              const bool invalidate_fetches, const bool invalidate_all) {
  for (std::size_t index = 0u; index < slot.undo.size(); ++index) {
    frames[slot.undo_frames[index]] = slot.undo[index];
  }
  if (invalidate_fetches) {
    for (const CacheBinding &binding : slot.bindings) {
      const bool uncertain_write =
          invalidate_all && (slot.ports.empty() || writes(binding.access));
      if ((uncertain_write || binding.fetch) && binding.frame < frames.size()) {
        frames[binding.frame] = frame_detail::empty(frames[binding.frame]);
      }
    }
    if (!slot.relocations.empty()) {
      for (const std::uint32_t index : slot.relocation_frames) {
        if (index >= frames.size()) {
          continue;
        }
        frames[index] = frame_detail::empty(frames[index]);
      }
    }
  }
}

[[nodiscard]] bool complete_epoch(std::vector<Authority::Frame> &frames,
                                  Authority::LeaseSlot &epoch,
                                  const bool success, const bool invalidate_all,
                                  CloseInfo *const info) noexcept {
  if (info != nullptr) {
    *info = CloseInfo{};
  }
  const auto reject = [&](const CloseInfo::Check check,
                          const std::uint32_t frame = CloseInfo::NoIndex,
                          const std::size_t binding_index =
                              std::numeric_limits<std::size_t>::max(),
                          const CacheBinding *const binding =
                              nullptr) noexcept {
    if (info != nullptr && info->check == CloseInfo::Check::None) {
      info->check = check;
      info->token = epoch.token;
      info->generation = epoch.generation;
      info->flags = static_cast<std::uint8_t>(
          (success ? CloseInfo::Success : CloseInfo::Flag{}) |
          (invalidate_all ? CloseInfo::Invalidate : CloseInfo::Flag{}));
      if (binding != nullptr) {
        info->binding_index =
            binding_index == std::numeric_limits<std::size_t>::max()
                ? CloseInfo::NoIndex
                : static_cast<std::uint32_t>(binding_index);
        info->binding_key = binding->key;
        info->binding_access = binding->access;
        info->binding_prior_dirty = binding->prior_dirty;
        info->binding_next_use = binding->next_use;
        info->binding_retain_until = binding->retain_until;
        info->binding_fetch = binding->fetch;
        info->binding_relocated = binding->relocated;
        info->binding_retire = binding->retire_on_success;
        info->binding_dirty = binding->dirty;
        info->flags = static_cast<std::uint8_t>(
            info->flags |
            (binding->fetch ? CloseInfo::Fetch : CloseInfo::Flag{}) |
            (binding->relocated ? CloseInfo::Relocated : CloseInfo::Flag{}) |
            (binding->retire_on_success ? CloseInfo::Retire
                                        : CloseInfo::Flag{}));
      }
      info->frame_index = frame;
      if (frame < frames.size()) {
        const Authority::Frame &value = frames[frame];
        info->frame_key = value.key;
        info->frame_assigned = value.assigned;
        info->frame_state = value.state;
        info->frame_tier = value.tier;
        info->frame_role = value.role;
        info->extent = value.extent;
        info->view = value.view;
        info->claims = value.alias_claims;
        info->frame_dirty = value.dirty;
        info->frame_next_use = value.next_use;
        info->frame_retain_until = value.retain_until;
      }
    }
    return false;
  };
  const auto same_identity = [](const Authority::Frame &left,
                                const Authority::Frame &right) noexcept {
    return left.assigned && right.assigned && left.key == right.key &&
           left.extent == right.extent && left.view == right.view;
  };
  const auto prior_for =
      [&epoch](const std::uint32_t value) noexcept -> const Authority::Frame * {
    for (std::size_t index = 0u; index < epoch.undo_frames.size(); ++index) {
      if (epoch.undo_frames[index] == value) {
        return &epoch.undo[index];
      }
    }
    return nullptr;
  };
  const auto claimed_read_safe =
      [&](const std::uint32_t value,
          const CacheBinding *const binding) noexcept {
        if (binding != nullptr &&
            (binding->access != Access::Read || binding->fetch ||
             binding->relocated || binding->retire_on_success ||
             !binding->dirty.empty() || !binding->prior_dirty.empty())) {
          return false;
        }
        for (const CacheBinding &candidate : epoch.bindings) {
          if (candidate.frame == value &&
              (candidate.access != Access::Read || candidate.fetch ||
               candidate.relocated || candidate.retire_on_success ||
               !candidate.dirty.empty() || !candidate.prior_dirty.empty())) {
            return false;
          }
        }
        if (std::find(epoch.relocation_frames.begin(),
                      epoch.relocation_frames.end(),
                      value) != epoch.relocation_frames.end()) {
          return false;
        }
        if (value >= frames.size()) {
          return false;
        }
        const Authority::Frame *const prior = prior_for(value);
        const Authority::Frame &current = frames[value];
        return prior != nullptr && prior->state == FrameState::Resident &&
               prior->dirty.empty() && same_identity(*prior, current) &&
               current.dirty.empty() &&
               (current.state == FrameState::Resident ||
                current.state == FrameState::Pinned ||
                current.state == FrameState::Mapping);
      };
  if (epoch.undo_frames.size() != epoch.undo.size()) {
    return reject(CloseInfo::Check::Undo);
  }
  for (std::size_t index = 0u; index < epoch.undo_frames.size(); ++index) {
    const std::uint32_t value = epoch.undo_frames[index];
    if (value >= frames.size() || (frames[value].alias_claims != 0u &&
                                   !claimed_read_safe(value, nullptr))) {
      return reject(value < frames.size() && frames[value].alias_claims != 0u
                        ? CloseInfo::Check::Alias
                        : CloseInfo::Check::Bounds,
                    value);
    }
  }
  if (!success) {
    for (std::size_t index = 0u; index < epoch.bindings.size(); ++index) {
      const CacheBinding &binding = epoch.bindings[index];
      const bool invalidate =
          binding.fetch ||
          (invalidate_all && (epoch.ports.empty() || writes(binding.access)));
      if (invalidate && (binding.frame >= frames.size() ||
                         frames[binding.frame].alias_claims != 0u)) {
        return reject(binding.frame < frames.size() &&
                              frames[binding.frame].alias_claims != 0u
                          ? CloseInfo::Check::Alias
                          : CloseInfo::Check::Bounds,
                      binding.frame, index, &binding);
      }
    }
    for (const std::uint32_t value : epoch.relocation_frames) {
      if (value >= frames.size() || frames[value].alias_claims != 0u) {
        return reject(value < frames.size() && frames[value].alias_claims != 0u
                          ? CloseInfo::Check::Alias
                          : CloseInfo::Check::Relocation,
                      value);
      }
    }
  }
  if (!success) {
    rollback(frames, epoch, true, invalidate_all);
  } else {
    if (!epoch.ports.empty() &&
        epoch.state != Authority::LeaseState::Computing) {
      return reject(CloseInfo::Check::State);
    }
    for (std::size_t index = 0u; index < epoch.bindings.size(); ++index) {
      const CacheBinding &binding = epoch.bindings[index];
      if (binding.frame >= frames.size()) {
        return reject(CloseInfo::Check::Bounds, binding.frame, index, &binding);
      }
      for (std::size_t prior = 0u; prior < index; ++prior) {
        if (epoch.bindings[prior].frame == binding.frame) {
          return reject(CloseInfo::Check::DuplicateFrame, binding.frame, index,
                        &binding);
        }
      }
      const Authority::Frame &frame = frames[binding.frame];
      if (frame.alias_claims != 0u &&
          !claimed_read_safe(binding.frame, &binding)) {
        return reject(CloseInfo::Check::Alias, binding.frame, index, &binding);
      }
      if (frame.key != binding.key ||
          (frame.state != FrameState::Pinned &&
           frame.state != FrameState::Mapping) ||
          (binding.retire_on_success && binding.access != Access::Read) ||
          (binding.retire_on_success &&
           binding.key.domain != CacheDomain::Transient) ||
          (binding.retire_on_success && frame.dirty.empty())) {
        if (frame.key != binding.key) {
          return reject(CloseInfo::Check::Key, binding.frame, index, &binding);
        }
        if (frame.state != FrameState::Pinned &&
            frame.state != FrameState::Mapping) {
          return reject(CloseInfo::Check::State, binding.frame, index,
                        &binding);
        }
        if (binding.retire_on_success && binding.access != Access::Read) {
          return reject(CloseInfo::Check::RetireAccess, binding.frame, index,
                        &binding);
        }
        if (binding.retire_on_success &&
            binding.key.domain != CacheDomain::Transient) {
          return reject(CloseInfo::Check::RetireDomain, binding.frame, index,
                        &binding);
        }
        return reject(CloseInfo::Check::RetirePayload, binding.frame, index,
                      &binding);
      }
    }
    for (const CacheBinding &binding : epoch.bindings) {
      Authority::Frame &frame = frames[binding.frame];
      if (binding.retire_on_success) {
        frame = frame_detail::empty(frame);
        continue;
      }
      if (writes(binding.access)) {
        frame.dirty = binding.dirty;
      }
      frame.state =
          frame.dirty.empty() ? FrameState::Resident : FrameState::Dirty;
    }
  }
  clear(epoch);
  return true;
}

} // namespace rund::compute::detail::residency
