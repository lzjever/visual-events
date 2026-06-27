#pragma once

#include "visual_events/dds_bridge/runtime_options.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace visual_events {
namespace dds_bridge {
namespace pc_test_tools {

struct JpegDimensions {
    uint32_t width = 0;
    uint32_t height = 0;
};

class UnitreeFactoryScope {
public:
    explicit UnitreeFactoryScope(const RuntimeOptions& options);
    UnitreeFactoryScope(const UnitreeFactoryScope&) = delete;
    UnitreeFactoryScope& operator=(const UnitreeFactoryScope&) = delete;
    ~UnitreeFactoryScope();

private:
    bool initialized_ = false;
};

std::string JsonEscape(std::string_view value);
int EmitFatal(std::string_view code, std::string_view message, int exit_code = 1);

bool ParsePositiveInt64(std::string_view text, std::string_view field, int64_t* out, std::string* error);
bool ParseNonNegativeInt64(
    std::string_view text,
    std::string_view field,
    int64_t* out,
    std::string* error);
bool ParsePositiveDouble(
    std::string_view text,
    std::string_view field,
    double* out,
    std::string* error);

int64_t SystemNowMs();
uint64_t MonotonicNowUint64Ns();
void SleepForHz(double hz);

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path);
std::vector<std::filesystem::path> ExpandJpegInput(const std::filesystem::path& input);
JpegDimensions ParseJpegDimensions(const std::vector<uint8_t>& bytes);

}  // namespace pc_test_tools
}  // namespace dds_bridge
}  // namespace visual_events
