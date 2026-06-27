#include "visual_events/dds_bridge/runtime_options.hpp"

#include "visual_events/dds_bridge/bridge_contract.hpp"

#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace visual_events {
namespace dds_bridge {
namespace {

constexpr const char* kDomainEnv = "VISUAL_EVENTS_DDS_DOMAIN";
constexpr const char* kNetworkEnv = "VISUAL_EVENTS_DDS_NETWORK";
constexpr const char* kCameraTopicEnv = "VISUAL_EVENTS_CAMERA_TOPIC";
constexpr const char* kHeadStateTopicEnv = "VISUAL_EVENTS_HEAD_STATE_TOPIC";
constexpr const char* kGazeTopicEnv = "VISUAL_EVENTS_GAZE_TOPIC";

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

std::string EnvOrDefault(const char* name, std::string_view default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return std::string(default_value);
    }
    return std::string(value);
}

bool ParseDomain(const char* value, int32_t* domain, std::string* error) {
    if (value == nullptr) {
        *domain = 0;
        return true;
    }
    if (*value == '\0') {
        *error = std::string(kDomainEnv) + " must be a non-negative integer";
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE || parsed < 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        *error = std::string(kDomainEnv) + " must be a non-negative integer";
        return false;
    }

    *domain = static_cast<int32_t>(parsed);
    return true;
}

bool RequireNonEmpty(std::string_view name, const std::string& value, std::string* error) {
    if (!value.empty()) {
        return true;
    }
    *error = std::string(name) + " must be non-empty";
    return false;
}

}  // namespace

RuntimeOptionsResult RuntimeOptionsFromEnv() {
    RuntimeOptionsResult result;
    if (!ParseDomain(std::getenv(kDomainEnv), &result.options.domain, &result.error)) {
        return result;
    }

    result.options.network = EnvOrDefault(kNetworkEnv, "eth0");
    result.options.camera_topic = EnvOrDefault(kCameraTopicEnv, kCameraTopic.name);
    result.options.head_state_topic = EnvOrDefault(kHeadStateTopicEnv, kHeadTopic.name);
    result.options.gaze_topic = EnvOrDefault(kGazeTopicEnv, kGazeTopic.name);

    if (!RequireNonEmpty(kNetworkEnv, result.options.network, &result.error) ||
        !RequireNonEmpty(kCameraTopicEnv, result.options.camera_topic, &result.error) ||
        !RequireNonEmpty(kHeadStateTopicEnv, result.options.head_state_topic, &result.error) ||
        !RequireNonEmpty(kGazeTopicEnv, result.options.gaze_topic, &result.error)) {
        return result;
    }

    result.ok = true;
    return result;
}

std::string EncodeRuntimeOptionsStatus(const RuntimeOptions& options, std::string_view mode) {
    std::ostringstream out;
    out << "{\"protocol_version\":" << kProtocolVersion
        << ",\"type\":\"status\""
        << ",\"code\":\"options_ok\""
        << ",\"message\":\"native Unitree channel construction options ok\""
        << ",\"mode\":" << JsonEscape(mode)
        << ",\"domain\":" << options.domain
        << ",\"network\":" << JsonEscape(options.network)
        << ",\"camera_topic\":" << JsonEscape(options.camera_topic)
        << ",\"head_state_topic\":" << JsonEscape(options.head_state_topic)
        << ",\"gaze_topic\":" << JsonEscape(options.gaze_topic)
        << "}";
    return out.str();
}

}  // namespace dds_bridge
}  // namespace visual_events
