#include "visual_events/dds_bridge/bridge_abi.hpp"

#include "visual_events/dds_bridge/bridge_contract.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace visual_events {
namespace dds_bridge {
namespace {

enum class JsonType {
    String,
    Integer,
    Number,
    Boolean,
};

struct JsonValue {
    JsonType type;
    std::string string_value;
    int64_t integer_value = 0;
    double number_value = 0.0;
    bool bool_value = false;
};

class JsonObjectParser {
public:
    explicit JsonObjectParser(std::string_view input) : input_(input) {}

    bool Parse(std::map<std::string, JsonValue>* out, std::string* error) {
        SkipWs();
        if (!Consume('{')) {
            *error = "expected JSON object";
            return false;
        }
        SkipWs();
        if (Consume('}')) {
            return true;
        }

        while (true) {
            std::string key;
            if (!ParseString(&key, error)) {
                return false;
            }
            SkipWs();
            if (!Consume(':')) {
                *error = "expected ':' after object key";
                return false;
            }
            SkipWs();
            JsonValue value;
            if (!ParseValue(&value, error)) {
                return false;
            }
            if (!out->emplace(key, value).second) {
                *error = "duplicate field: " + key;
                return false;
            }
            SkipWs();
            if (Consume('}')) {
                break;
            }
            if (!Consume(',')) {
                *error = "expected ',' or '}'";
                return false;
            }
            SkipWs();
        }

        SkipWs();
        if (pos_ != input_.size()) {
            *error = "trailing data after JSON object";
            return false;
        }
        return true;
    }

private:
    void SkipWs() {
        while (pos_ < input_.size()) {
            const char c = input_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                return;
            }
            ++pos_;
        }
    }

    bool Consume(char expected) {
        if (pos_ >= input_.size() || input_[pos_] != expected) {
            return false;
        }
        ++pos_;
        return true;
    }

    bool ParseValue(JsonValue* value, std::string* error) {
        if (pos_ >= input_.size()) {
            *error = "unexpected end of JSON value";
            return false;
        }
        const char c = input_[pos_];
        if (c == '"') {
            value->type = JsonType::String;
            return ParseString(&value->string_value, error);
        }
        if (c == 't' || c == 'f') {
            value->type = JsonType::Boolean;
            return ParseBoolean(&value->bool_value, error);
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return ParseNumber(value, error);
        }
        *error = "unsupported JSON value";
        return false;
    }

    bool ParseString(std::string* out, std::string* error) {
        if (!Consume('"')) {
            *error = "expected string";
            return false;
        }
        out->clear();
        while (pos_ < input_.size()) {
            const unsigned char c = static_cast<unsigned char>(input_[pos_++]);
            if (c == '"') {
                return true;
            }
            if (c < 0x20) {
                *error = "unescaped control character in string";
                return false;
            }
            if (c != '\\') {
                out->push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= input_.size()) {
                *error = "unterminated escape in string";
                return false;
            }
            const char escaped = input_[pos_++];
            switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    out->push_back(escaped);
                    break;
                case 'b':
                    out->push_back('\b');
                    break;
                case 'f':
                    out->push_back('\f');
                    break;
                case 'n':
                    out->push_back('\n');
                    break;
                case 'r':
                    out->push_back('\r');
                    break;
                case 't':
                    out->push_back('\t');
                    break;
                case 'u':
                    if (!ParseUnicodeEscape(out, error)) {
                        return false;
                    }
                    break;
                default:
                    *error = "invalid string escape";
                    return false;
            }
        }
        *error = "unterminated string";
        return false;
    }

    bool ParseUnicodeEscape(std::string* out, std::string* error) {
        if (pos_ + 4 > input_.size()) {
            *error = "short unicode escape";
            return false;
        }
        int value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value += c - '0';
            } else if (c >= 'a' && c <= 'f') {
                value += c - 'a' + 10;
            } else if (c >= 'A' && c <= 'F') {
                value += c - 'A' + 10;
            } else {
                *error = "invalid unicode escape";
                return false;
            }
        }

        if (value <= 0x7F) {
            out->push_back(static_cast<char>(value));
        } else if (value <= 0x7FF) {
            out->push_back(static_cast<char>(0xC0 | ((value >> 6) & 0x1F)));
            out->push_back(static_cast<char>(0x80 | (value & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xE0 | ((value >> 12) & 0x0F)));
            out->push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (value & 0x3F)));
        }
        return true;
    }

    bool ParseBoolean(bool* out, std::string* error) {
        if (input_.substr(pos_, 4) == "true") {
            pos_ += 4;
            *out = true;
            return true;
        }
        if (input_.substr(pos_, 5) == "false") {
            pos_ += 5;
            *out = false;
            return true;
        }
        *error = "invalid boolean";
        return false;
    }

    bool ParseNumber(JsonValue* value, std::string* error) {
        const size_t start = pos_;
        if (input_[pos_] == '-') {
            ++pos_;
            if (pos_ >= input_.size()) {
                *error = "invalid number";
                return false;
            }
        }
        if (input_[pos_] == '0') {
            ++pos_;
        } else if (input_[pos_] >= '1' && input_[pos_] <= '9') {
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            *error = "invalid number";
            return false;
        }

        bool integer = true;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            integer = false;
            ++pos_;
            if (pos_ >= input_.size() || input_[pos_] < '0' || input_[pos_] > '9') {
                *error = "invalid number fraction";
                return false;
            }
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                ++pos_;
            }
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            integer = false;
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= input_.size() || input_[pos_] < '0' || input_[pos_] > '9') {
                *error = "invalid number exponent";
                return false;
            }
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') {
                ++pos_;
            }
        }

        const std::string raw(input_.substr(start, pos_ - start));
        if (integer) {
            try {
                size_t consumed = 0;
                const long long parsed = std::stoll(raw, &consumed, 10);
                if (consumed != raw.size()) {
                    *error = "invalid integer";
                    return false;
                }
                value->type = JsonType::Integer;
                value->integer_value = static_cast<int64_t>(parsed);
                value->number_value = static_cast<double>(parsed);
                return true;
            } catch (const std::exception&) {
                *error = "integer out of range";
                return false;
            }
        }

        errno = 0;
        char* end = nullptr;
        const double parsed = std::strtod(raw.c_str(), &end);
        if (end == raw.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(parsed)) {
            *error = "number must be finite";
            return false;
        }
        value->type = JsonType::Number;
        value->number_value = parsed;
        return true;
    }

    std::string_view input_;
    size_t pos_ = 0;
};

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

std::string Base64Encode(const std::vector<uint8_t>& data) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
        const uint32_t b2 = (i + 2 < data.size()) ? data[i + 2] : 0;
        const uint32_t packed = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(packed >> 18) & 0x3F]);
        out.push_back(kAlphabet[(packed >> 12) & 0x3F]);
        out.push_back((i + 1 < data.size()) ? kAlphabet[(packed >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < data.size()) ? kAlphabet[packed & 0x3F] : '=');
    }
    return out;
}

bool IsAllowedHeadState(std::string_view state) {
    return state == "stationary" || state == "moving" || state == "unknown";
}

bool IsAllowedGazeTargetState(std::string_view state) {
    return state == "tracking" || state == "lost" || state == "stale" || state == "disabled";
}

bool GetString(
    const std::map<std::string, JsonValue>& object,
    const std::string& field,
    std::string* out,
    std::string* error) {
    const auto it = object.find(field);
    if (it == object.end()) {
        *error = "missing field: " + field;
        return false;
    }
    if (it->second.type != JsonType::String) {
        *error = field + " must be a string";
        return false;
    }
    *out = it->second.string_value;
    return true;
}

bool GetInteger(
    const std::map<std::string, JsonValue>& object,
    const std::string& field,
    int64_t* out,
    std::string* error) {
    const auto it = object.find(field);
    if (it == object.end()) {
        *error = "missing field: " + field;
        return false;
    }
    if (it->second.type != JsonType::Integer) {
        *error = field + " must be an integer";
        return false;
    }
    *out = it->second.integer_value;
    return true;
}

bool GetBool(
    const std::map<std::string, JsonValue>& object,
    const std::string& field,
    bool* out,
    std::string* error) {
    const auto it = object.find(field);
    if (it == object.end()) {
        *error = "missing field: " + field;
        return false;
    }
    if (it->second.type != JsonType::Boolean) {
        *error = field + " must be a boolean";
        return false;
    }
    *out = it->second.bool_value;
    return true;
}

bool GetFiniteNumber(
    const std::map<std::string, JsonValue>& object,
    const std::string& field,
    double* out,
    std::string* error) {
    const auto it = object.find(field);
    if (it == object.end()) {
        *error = "missing field: " + field;
        return false;
    }
    if (it->second.type != JsonType::Integer && it->second.type != JsonType::Number) {
        *error = field + " must be a number";
        return false;
    }
    const double value = it->second.number_value;
    if (!std::isfinite(value)) {
        *error = field + " must be finite";
        return false;
    }
    *out = value;
    return true;
}

std::string EncodeDouble(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

}  // namespace

int64_t MonotonicNowNs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::string EncodeStatusFrame(
    std::string_view code,
    std::string_view message,
    std::string_view mode) {
    std::ostringstream out;
    out << "{\"protocol_version\":" << kProtocolVersion
        << ",\"type\":\"status\""
        << ",\"code\":" << JsonEscape(code)
        << ",\"message\":" << JsonEscape(message);
    if (!mode.empty()) {
        out << ",\"mode\":" << JsonEscape(mode);
    }
    out << "}";
    return out.str();
}

std::string EncodeErrorFrame(std::string_view code, std::string_view message, bool fatal) {
    std::ostringstream out;
    out << "{\"protocol_version\":" << kProtocolVersion
        << ",\"type\":\"error\""
        << ",\"code\":" << JsonEscape(code)
        << ",\"message\":" << JsonEscape(message)
        << ",\"fatal\":" << (fatal ? "true" : "false")
        << "}";
    return out.str();
}

std::string EncodeCameraJpegFrame(const CameraJpegFrame& frame) {
    std::ostringstream out;
    out << "{\"protocol_version\":" << kProtocolVersion
        << ",\"type\":\"camera_jpeg\""
        << ",\"dds_timestamp_ns\":" << frame.dds_timestamp_ns
        << ",\"received_monotonic_ns\":" << frame.received_monotonic_ns
        << ",\"camera_name\":" << JsonEscape(frame.camera_name)
        << ",\"width\":" << frame.width
        << ",\"height\":" << frame.height
        << ",\"encoding\":" << JsonEscape(frame.encoding)
        << ",\"step\":" << frame.step
        << ",\"data_size_bytes\":" << frame.data.size()
        << ",\"data_base64\":" << JsonEscape(Base64Encode(frame.data))
        << "}";
    return out.str();
}

std::string EncodeHeadStateFrame(const HeadStateFrame& frame) {
    const std::string state = IsAllowedHeadState(frame.state) ? frame.state : "unknown";
    std::ostringstream out;
    out << "{\"protocol_version\":" << kProtocolVersion
        << ",\"type\":\"head_state\""
        << ",\"dds_timestamp_ns\":" << frame.dds_timestamp_ns
        << ",\"received_monotonic_ns\":" << frame.received_monotonic_ns
        << ",\"valid\":" << (frame.valid ? "true" : "false")
        << ",\"state\":" << JsonEscape(state)
        << ",\"yaw_rad\":" << EncodeDouble(std::isfinite(frame.yaw_rad) ? frame.yaw_rad : 0.0)
        << ",\"pitch_rad\":" << EncodeDouble(std::isfinite(frame.pitch_rad) ? frame.pitch_rad : 0.0)
        << ",\"yaw_vel_rad_s\":"
        << EncodeDouble(std::isfinite(frame.yaw_vel_rad_s) ? frame.yaw_vel_rad_s : 0.0)
        << ",\"pitch_vel_rad_s\":"
        << EncodeDouble(std::isfinite(frame.pitch_vel_rad_s) ? frame.pitch_vel_rad_s : 0.0)
        << "}";
    return out.str();
}

GazeTargetParseResult ParseGazeTargetLine(std::string_view line) {
    GazeTargetParseResult result;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.remove_suffix(1);
    }
    if (line.find('\n') != std::string_view::npos || line.find('\r') != std::string_view::npos) {
        result.error = "gaze_target must be one JSONL frame";
        return result;
    }

    std::map<std::string, JsonValue> object;
    JsonObjectParser parser(line);
    if (!parser.Parse(&object, &result.error)) {
        return result;
    }

    static const char* const kFields[] = {
        "protocol_version",
        "type",
        "schema_version",
        "camera",
        "frame_id",
        "frame_timestamp_ms",
        "publish_timestamp_ms",
        "valid",
        "state",
        "target_track_id",
        "target_u",
        "target_v",
        "target_norm_x",
        "target_norm_y",
        "image_width",
        "image_height",
        "confidence",
        "reason",
        "stale_after_ms",
    };
    if (object.size() != (sizeof(kFields) / sizeof(kFields[0]))) {
        result.error = "gaze_target must contain exactly the canonical fields";
        return result;
    }
    for (const char* field : kFields) {
        if (object.find(field) == object.end()) {
            result.error = std::string("missing field: ") + field;
            return result;
        }
    }

    int64_t protocol_version = 0;
    std::string type;
    if (!GetInteger(object, "protocol_version", &protocol_version, &result.error) ||
        !GetString(object, "type", &type, &result.error)) {
        return result;
    }
    if (protocol_version != kProtocolVersion) {
        result.error = "unsupported protocol_version";
        return result;
    }
    if (type != "gaze_target") {
        result.error = "type must be gaze_target";
        return result;
    }

    GazeTargetFrame frame;
    if (!GetInteger(object, "schema_version", &frame.schema_version, &result.error) ||
        !GetString(object, "camera", &frame.camera, &result.error) ||
        !GetInteger(object, "frame_id", &frame.frame_id, &result.error) ||
        !GetInteger(object, "frame_timestamp_ms", &frame.frame_timestamp_ms, &result.error) ||
        !GetInteger(object, "publish_timestamp_ms", &frame.publish_timestamp_ms, &result.error) ||
        !GetBool(object, "valid", &frame.valid, &result.error) ||
        !GetString(object, "state", &frame.state, &result.error) ||
        !GetInteger(object, "target_track_id", &frame.target_track_id, &result.error) ||
        !GetFiniteNumber(object, "target_u", &frame.target_u, &result.error) ||
        !GetFiniteNumber(object, "target_v", &frame.target_v, &result.error) ||
        !GetFiniteNumber(object, "target_norm_x", &frame.target_norm_x, &result.error) ||
        !GetFiniteNumber(object, "target_norm_y", &frame.target_norm_y, &result.error) ||
        !GetInteger(object, "image_width", &frame.image_width, &result.error) ||
        !GetInteger(object, "image_height", &frame.image_height, &result.error) ||
        !GetFiniteNumber(object, "confidence", &frame.confidence, &result.error) ||
        !GetString(object, "reason", &frame.reason, &result.error) ||
        !GetInteger(object, "stale_after_ms", &frame.stale_after_ms, &result.error)) {
        return result;
    }
    if (!IsAllowedGazeTargetState(frame.state)) {
        result.error = "state must be tracking, lost, stale, or disabled";
        return result;
    }
    result.ok = true;
    result.frame = frame;
    return result;
}

}  // namespace dds_bridge
}  // namespace visual_events
