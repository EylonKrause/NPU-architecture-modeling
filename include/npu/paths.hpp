#pragma once

#include <filesystem>
#include <system_error>

namespace npu {

// True when both name one file: the same filesystem object (hard links
// included), or the same weakly canonical path, which also matches a final
// component that does not exist yet.
inline bool same_file_path(const std::filesystem::path& left, const std::filesystem::path& right) {
    std::error_code error;
    if (std::filesystem::equivalent(left, right, error)) return true;
    return std::filesystem::weakly_canonical(std::filesystem::absolute(left)) ==
           std::filesystem::weakly_canonical(std::filesystem::absolute(right));
}

}  // namespace npu
