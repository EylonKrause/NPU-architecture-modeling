#pragma once

#include <filesystem>
#include <system_error>

namespace npu {

// Compare both existing filesystem identities (including hard links) and paths
// whose final file has not been created yet. Weak canonicalization resolves
// symlinked parent directories while retaining a nonexistent final component.
inline bool same_file_path(const std::filesystem::path& left, const std::filesystem::path& right) {
    std::error_code error;
    if (std::filesystem::equivalent(left, right, error)) return true;
    return std::filesystem::weakly_canonical(std::filesystem::absolute(left)) ==
           std::filesystem::weakly_canonical(std::filesystem::absolute(right));
}

}  // namespace npu
