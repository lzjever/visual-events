#include "visual_events/dds_bridge/pc_test_tools.hpp"

#include "visual_events/dds_bridge/bridge_abi.hpp"

#include "unitree/robot/channel/channel_factory.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace visual_events {
namespace dds_bridge {
namespace pc_test_tools {
namespace {

bool IsJpegExtension(std::string extension) {
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".jpg" || extension == ".jpeg";
}

uint16_t ReadBigEndianU16(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint16_t>((static_cast<uint16_t>(bytes[offset]) << 8U) | bytes[offset + 1]);
}

bool IsSofMarker(uint8_t marker) {
    return (marker >= 0xC0 && marker <= 0xC3) || (marker >= 0xC5 && marker <= 0xC7) ||
           (marker >= 0xC9 && marker <= 0xCB) || (marker >= 0xCD && marker <= 0xCF);
}

bool IsStandaloneMarker(uint8_t marker) {
    return marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9);
}

}  // namespace

UnitreeFactoryScope::UnitreeFactoryScope(const RuntimeOptions& options) {
    unitree::robot::ChannelFactory::Instance()->Init(options.domain, options.network);
    initialized_ = true;
}

UnitreeFactoryScope::~UnitreeFactoryScope() {
    if (initialized_) {
        unitree::robot::ChannelFactory::Instance()->Release();
    }
}

std::string JsonEscape(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
                break;
        }
    }
    out << '"';
    return out.str();
}

int EmitFatal(std::string_view code, std::string_view message, int exit_code) {
    std::cerr << message << '\n';
    std::cout << EncodeErrorFrame(code, message, true) << '\n';
    std::cout.flush();
    return exit_code;
}

bool ParsePositiveInt64(std::string_view text, std::string_view field, int64_t* out, std::string* error) {
    int64_t parsed = 0;
    if (!ParseNonNegativeInt64(text, field, &parsed, error) || parsed <= 0) {
        if (error != nullptr) {
            *error = std::string(field) + " must be a positive integer";
        }
        return false;
    }
    *out = parsed;
    return true;
}

bool ParseNonNegativeInt64(
    std::string_view text,
    std::string_view field,
    int64_t* out,
    std::string* error) {
    if (text.empty()) {
        if (error != nullptr) {
            *error = std::string(field) + " must be a non-negative integer";
        }
        return false;
    }
    const std::string raw(text);
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(raw.c_str(), &end, 10);
    if (end == raw.c_str() || *end != '\0' || errno == ERANGE || parsed < 0) {
        if (error != nullptr) {
            *error = std::string(field) + " must be a non-negative integer";
        }
        return false;
    }
    *out = static_cast<int64_t>(parsed);
    return true;
}

bool ParsePositiveDouble(
    std::string_view text,
    std::string_view field,
    double* out,
    std::string* error) {
    if (text.empty()) {
        if (error != nullptr) {
            *error = std::string(field) + " must be a positive number";
        }
        return false;
    }
    const std::string raw(text);
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(raw.c_str(), &end);
    if (end == raw.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(parsed) ||
        parsed <= 0.0) {
        if (error != nullptr) {
            *error = std::string(field) + " must be a positive number";
        }
        return false;
    }
    *out = parsed;
    return true;
}

int64_t SystemNowMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

uint64_t MonotonicNowUint64Ns() {
    const int64_t now = MonotonicNowNs();
    if (now < 0) {
        return 0;
    }
    return static_cast<uint64_t>(now);
}

void SleepForHz(double hz) {
    if (hz <= 0.0 || !std::isfinite(hz)) {
        return;
    }
    const auto delay = std::chrono::duration<double>(1.0 / hz);
    std::this_thread::sleep_for(delay);
}

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input JPEG: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to size input JPEG: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw std::runtime_error("failed to read input JPEG: " + path.string());
        }
    }
    return bytes;
}

std::vector<std::filesystem::path> ExpandJpegInput(const std::filesystem::path& input) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(input, ec)) {
        return {input};
    }
    if (ec) {
        throw std::runtime_error("failed to inspect input path: " + input.string());
    }
    if (!std::filesystem::is_directory(input, ec)) {
        throw std::runtime_error("input path is not a file or directory: " + input.string());
    }
    if (ec) {
        throw std::runtime_error("failed to inspect input directory: " + input.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
        if (entry.is_regular_file() && IsJpegExtension(entry.path().extension().string())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("input directory contains no .jpg or .jpeg files: " + input.string());
    }
    return paths;
}

JpegDimensions ParseJpegDimensions(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8) {
        throw std::runtime_error("JPEG payload does not start with SOI");
    }

    size_t pos = 2;
    while (pos < bytes.size()) {
        if (bytes[pos] != 0xFF) {
            throw std::runtime_error("malformed JPEG marker stream before SOF");
        }
        while (pos < bytes.size() && bytes[pos] == 0xFF) {
            ++pos;
        }
        if (pos >= bytes.size()) {
            break;
        }

        const uint8_t marker = bytes[pos++];
        if (IsStandaloneMarker(marker)) {
            if (marker == 0xD9) {
                break;
            }
            continue;
        }
        if (pos + 2 > bytes.size()) {
            throw std::runtime_error("truncated JPEG segment length");
        }
        const uint16_t segment_length = ReadBigEndianU16(bytes, pos);
        if (segment_length < 2 || pos + segment_length > bytes.size()) {
            throw std::runtime_error("invalid JPEG segment length");
        }
        if (marker == 0xDA) {
            break;
        }
        if (IsSofMarker(marker)) {
            if (segment_length < 7) {
                throw std::runtime_error("truncated JPEG SOF segment");
            }
            const uint16_t height = ReadBigEndianU16(bytes, pos + 3);
            const uint16_t width = ReadBigEndianU16(bytes, pos + 5);
            if (width == 0 || height == 0) {
                throw std::runtime_error("JPEG SOF dimensions must be non-zero");
            }
            return JpegDimensions{width, height};
        }
        pos += segment_length;
    }

    throw std::runtime_error("JPEG SOF dimensions not found");
}

}  // namespace pc_test_tools
}  // namespace dds_bridge
}  // namespace visual_events
