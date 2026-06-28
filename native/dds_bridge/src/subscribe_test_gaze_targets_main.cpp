#include "visual_events/dds_bridge/pc_test_tools.hpp"
#include "visual_events/dds_bridge/runtime_options.hpp"

#include "gaze_target_v1.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace {

enum class CollectionMode {
    kCount,
    kDuration,
};

struct Args {
    int64_t count = 0;
    int64_t timeout_ms = -1;
    int64_t duration_ms = 0;
    int64_t min_count = 0;
    CollectionMode mode = CollectionMode::kCount;
};

struct State {
    std::mutex mutex;
    std::condition_variable cv;
    int64_t received = 0;
    bool failed = false;
    std::string error;
};

constexpr const char* kUsage =
    "usage: visual_events_dds_bridge_subscribe_test_gaze_targets "
    "(--count <n> --timeout-ms <ms> | --duration-ms <ms> --min-count <n>)";

bool NeedValue(int index, int argc, const char* flag, std::string* error) {
    if (index + 1 < argc) {
        return true;
    }
    *error = std::string(flag) + " requires a value";
    return false;
}

bool ParseArgs(int argc, char** argv, Args* args, std::string* error) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--count") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            if (!visual_events::dds_bridge::pc_test_tools::ParsePositiveInt64(
                    argv[++i],
                    "--count",
                    &args->count,
                    error)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--timeout-ms") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            if (!visual_events::dds_bridge::pc_test_tools::ParseNonNegativeInt64(
                    argv[++i],
                    "--timeout-ms",
                    &args->timeout_ms,
                    error)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--duration-ms") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            if (!visual_events::dds_bridge::pc_test_tools::ParsePositiveInt64(
                    argv[++i],
                    "--duration-ms",
                    &args->duration_ms,
                    error)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--min-count") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            if (!visual_events::dds_bridge::pc_test_tools::ParsePositiveInt64(
                    argv[++i],
                    "--min-count",
                    &args->min_count,
                    error)) {
                return false;
            }
        } else {
            *error = std::string("unknown argument: ") + argv[i];
            return false;
        }
    }

    const bool count_mode = args->count > 0 || args->timeout_ms >= 0;
    const bool duration_mode = args->duration_ms > 0 || args->min_count > 0;
    if (count_mode == duration_mode) {
        *error = kUsage;
        return false;
    }
    if (count_mode) {
        if (args->count <= 0 || args->timeout_ms < 0) {
            *error = kUsage;
            return false;
        }
        args->mode = CollectionMode::kCount;
    } else {
        if (args->duration_ms <= 0 || args->min_count <= 0) {
            *error = kUsage;
            return false;
        }
        args->mode = CollectionMode::kDuration;
    }
    return true;
}

std::string EncodeFloat(float value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

std::string EncodeGazeTargetJsonl(const visual_events::msg::dds_::GazeTargetV1_& frame) {
    namespace tools = visual_events::dds_bridge::pc_test_tools;
    std::ostringstream out;
    out << "{\"protocol_version\":1"
        << ",\"type\":\"gaze_target\""
        << ",\"schema_version\":" << frame.schema_version()
        << ",\"camera\":" << tools::JsonEscape(frame.camera())
        << ",\"frame_id\":" << frame.frame_id()
        << ",\"frame_timestamp_ms\":" << frame.frame_timestamp_ms()
        << ",\"publish_timestamp_ms\":" << frame.publish_timestamp_ms()
        << ",\"valid\":" << (frame.valid() ? "true" : "false")
        << ",\"state\":" << tools::JsonEscape(frame.state())
        << ",\"target_track_id\":" << frame.target_track_id()
        << ",\"target_u\":" << EncodeFloat(frame.target_u())
        << ",\"target_v\":" << EncodeFloat(frame.target_v())
        << ",\"target_norm_x\":" << EncodeFloat(frame.target_norm_x())
        << ",\"target_norm_y\":" << EncodeFloat(frame.target_norm_y())
        << ",\"image_width\":" << frame.image_width()
        << ",\"image_height\":" << frame.image_height()
        << ",\"confidence\":" << EncodeFloat(frame.confidence())
        << ",\"reason\":" << tools::JsonEscape(frame.reason())
        << ",\"stale_after_ms\":" << frame.stale_after_ms()
        << "}";
    return out.str();
}

void OnGazeTargetSample(const void* sample, State* state, std::mutex* output_mutex) {
    if (sample == nullptr) {
        return;
    }
    try {
        const auto* frame = static_cast<const visual_events::msg::dds_::GazeTargetV1_*>(sample);
        const std::string line = EncodeGazeTargetJsonl(*frame);
        {
            std::lock_guard<std::mutex> output_lock(*output_mutex);
            std::cout << line << '\n';
            std::cout.flush();
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            ++state->received;
        }
        state->cv.notify_all();
    } catch (const std::exception& exc) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->failed = true;
            state->error = exc.what();
        }
        state->cv.notify_all();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->failed = true;
            state->error = "unknown gaze_target callback failure";
        }
        state->cv.notify_all();
    }
}

}  // namespace

int main(int argc, char** argv) {
    namespace tools = visual_events::dds_bridge::pc_test_tools;

    Args args;
    std::string error;
    if (!ParseArgs(argc, argv, &args, &error)) {
        if (error != kUsage) {
            error = error + "; " + kUsage;
        }
        return tools::EmitFatal("invalid_arguments", error, 2);
    }

    const auto parsed_options = visual_events::dds_bridge::RuntimeOptionsFromEnv();
    if (!parsed_options.ok) {
        return tools::EmitFatal("invalid_runtime_options", parsed_options.error);
    }

    try {
        tools::UnitreeFactoryScope factory(parsed_options.options);
        State state;
        std::mutex output_mutex;
        unitree::robot::ChannelSubscriber<visual_events::msg::dds_::GazeTargetV1_> subscriber(
            parsed_options.options.gaze_topic);
        subscriber.InitChannel([&state, &output_mutex](const void* sample) {
            OnGazeTargetSample(sample, &state, &output_mutex);
        });

        bool completed = false;
        int64_t received_count = 0;
        if (args.mode == CollectionMode::kCount) {
            std::unique_lock<std::mutex> lock(state.mutex);
            const auto done = [&state, &args] {
                return state.failed || state.received >= args.count;
            };
            if (args.timeout_ms == 0) {
                state.cv.wait(lock, done);
                completed = !state.failed;
            } else {
                completed = state.cv.wait_for(
                    lock,
                    std::chrono::milliseconds(args.timeout_ms),
                    done);
            }
            if (state.failed) {
                error = state.error;
                completed = false;
            }
            received_count = state.received;
        } else {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait_for(
                lock,
                std::chrono::milliseconds(args.duration_ms),
                [&state] { return state.failed; });
            if (state.failed) {
                error = state.error;
                completed = false;
            } else {
                completed = state.received >= args.min_count;
            }
            received_count = state.received;
        }

        subscriber.CloseChannel();
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (state.failed && error.empty()) {
                error = state.error;
                completed = false;
            }
            received_count = state.received;
        }
        if (!error.empty()) {
            return tools::EmitFatal("gaze_target_callback_failed", error);
        }
        if (!completed) {
            std::ostringstream message;
            if (args.mode == CollectionMode::kCount) {
                message << "timed out waiting for " << args.count
                        << " gaze_target sample(s) on "
                        << parsed_options.options.gaze_topic;
                return tools::EmitFatal("timeout", message.str());
            }
            message << "collected " << received_count << " gaze_target sample(s) in "
                    << args.duration_ms << "ms; expected at least " << args.min_count
                    << " on " << parsed_options.options.gaze_topic;
            return tools::EmitFatal("insufficient_samples", message.str());
        }
        return 0;
    } catch (const std::exception& exc) {
        return tools::EmitFatal("subscribe_test_gaze_targets_failed", exc.what());
    } catch (...) {
        return tools::EmitFatal(
            "subscribe_test_gaze_targets_failed",
            "unknown gaze_target subscribe failure");
    }
}
