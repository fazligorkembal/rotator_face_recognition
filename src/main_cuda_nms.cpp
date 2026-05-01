#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>

#include <json.hpp>

#include "detection_model_inference_helper.h"
#include "identification_model_inference_helper.h"
#include "tensorrt/logging.h"
#include "spd_logger_helper.h"
#include "utils.hpp"

using json = nlohmann::json;

namespace
{

std::atomic<bool> is_run{true};
Logger gLogger(Severity::kINFO);
constexpr int kFixedCaptureWidth = 1920;
constexpr int kFixedCaptureHeight = 1080;
constexpr int kFixedCaptureFps = 30;
constexpr int kCameraFlipMethod180 = 2;

const char *detectionTypeName(DetectionType detection_type)
{
    return detection_type == DetectionType::SEARCH ? "search" : "track";
}

cv::Rect makeCenteredRoiFromHalfExtents(int frame_width,
                                        int frame_height,
                                        int half_width,
                                        int half_height)
{
    const int roi_width = std::clamp(half_width * 2, 1, frame_width);
    const int roi_height = std::clamp(half_height * 2, 1, frame_height);
    const int x = std::max(0, (frame_width - roi_width) / 2);
    const int y = std::max(0, (frame_height - roi_height) / 2);
    return cv::Rect(x, y, roi_width, roi_height);
}

cv::Point detectionCenter(const DetectionBox &detection)
{
    return cv::Point((detection.x1 + detection.x2) / 2, (detection.y1 + detection.y2) / 2);
}

std::optional<DetectionBox> findBestDetectionInRoi(const std::vector<DetectionBox> &detections,
                                                   const cv::Rect &roi,
                                                   int frame_center_x,
                                                   int frame_center_y)
{
    std::optional<DetectionBox> best_detection;
    int64_t best_distance_sq = std::numeric_limits<int64_t>::max();

    for (const auto &detection : detections)
    {
        const cv::Point center = detectionCenter(detection);
        if (!roi.contains(center))
        {
            continue;
        }

        const int64_t dx = static_cast<int64_t>(center.x) - frame_center_x;
        const int64_t dy = static_cast<int64_t>(center.y) - frame_center_y;
        const int64_t distance_sq = dx * dx + dy * dy;
        if (!best_detection.has_value() ||
            distance_sq < best_distance_sq ||
            (distance_sq == best_distance_sq && detection.score > best_detection->score))
        {
            best_detection = detection;
            best_distance_sq = distance_sq;
        }
    }

    return best_detection;
}

std::vector<DetectionBox> mapTrackDetectionsToFrame(const std::vector<DetectionBox> &detections,
                                                    const cv::Rect &track_roi,
                                                    int model_width,
                                                    int model_height)
{
    std::vector<DetectionBox> adjusted;
    adjusted.reserve(detections.size());

    const double scale_x = static_cast<double>(track_roi.width) / std::max(1, model_width);
    const double scale_y = static_cast<double>(track_roi.height) / std::max(1, model_height);

    for (const auto &detection : detections)
    {
        DetectionBox mapped = detection;
        mapped.x1 = track_roi.x + static_cast<int>(std::lround(static_cast<double>(detection.x1) * scale_x));
        mapped.y1 = track_roi.y + static_cast<int>(std::lround(static_cast<double>(detection.y1) * scale_y));
        mapped.x2 = track_roi.x + static_cast<int>(std::lround(static_cast<double>(detection.x2) * scale_x));
        mapped.y2 = track_roi.y + static_cast<int>(std::lround(static_cast<double>(detection.y2) * scale_y));
        adjusted.push_back(mapped);
    }

    return adjusted;
}

class DetectionModeController
{
public:
    explicit DetectionModeController(cv::Rect track_roi)
        : track_roi_(std::move(track_roi))
    {
    }

    DetectionType requestedMode() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return requested_mode_;
    }

    void updateFromDetections(const std::vector<DetectionBox> &detections,
                              int frame_center_x,
                              int frame_center_y)
    {
        const auto best_target = findBestDetectionInRoi(detections, track_roi_, frame_center_x, frame_center_y);
        const DetectionType next_mode = best_target.has_value() ? DetectionType::TRACK : DetectionType::SEARCH;

        std::lock_guard<std::mutex> lock(mutex_);
        if (next_mode != requested_mode_)
        {
            LOG_INFO("AI mode switched to {}",
                     next_mode == DetectionType::SEARCH ? "SEARCH" : "TRACK");
        }
        requested_mode_ = next_mode;
    }

private:
    cv::Rect track_roi_;
    mutable std::mutex mutex_;
    DetectionType requested_mode_ = DetectionType::SEARCH;
};

void handleSignal(int)
{
    is_run.store(false);
}

std::filesystem::path resolveProjectRoot()
{
    auto current = std::filesystem::current_path();
    if (current.filename() == "build")
    {
        return current.parent_path();
    }
    if (std::filesystem::exists(current / "configs" / "calib.json"))
    {
        return current;
    }
    return current / "rotator_face_recognition";
}

std::string readTextFile(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string replaceAll(std::string text, const std::string &from, const std::string &to)
{
    size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos)
    {
        text.replace(position, from.size(), to);
        position += to.size();
    }
    return text;
}

std::string escapeHtml(const std::string &input)
{
    std::string escaped;
    escaped.reserve(input.size());
    for (char ch : input)
    {
        switch (ch)
        {
        case '&':
            escaped += "&amp;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        case '"':
            escaped += "&quot;";
            break;
        case '\'':
            escaped += "&#39;";
            break;
        default:
            escaped.push_back(ch);
            break;
        }
    }
    return escaped;
}

std::string escapeGStreamerString(const std::string &value)
{
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char ch : value)
    {
        if (ch == '\\' || ch == '"')
        {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

bool hasGStreamerElement(const char *name)
{
    GstElementFactory *factory = gst_element_factory_find(name);
    if (factory == nullptr)
    {
        return false;
    }

    gst_object_unref(factory);
    return true;
}

std::string chooseGStreamerElement(const std::vector<const char *> &candidates)
{
    for (const char *candidate : candidates)
    {
        if (hasGStreamerElement(candidate))
        {
            return candidate;
        }
    }
    return {};
}

std::string guessLocalIpAddress()
{
    struct ifaddrs *interfaces = nullptr;
    if (getifaddrs(&interfaces) != 0)
    {
        return "127.0.0.1";
    }

    std::string ip = "127.0.0.1";
    for (struct ifaddrs *ifa = interfaces; ifa != nullptr; ifa = ifa->ifa_next)
    {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET)
        {
            continue;
        }
        if (ifa->ifa_flags & IFF_LOOPBACK)
        {
            continue;
        }

        char address[INET_ADDRSTRLEN] = {};
        auto *sock_addr = reinterpret_cast<sockaddr_in *>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &sock_addr->sin_addr, address, sizeof(address)) != nullptr)
        {
            ip = address;
            break;
        }
    }

    freeifaddrs(interfaces);
    return ip;
}

bool writeAll(int fd, const void *data, size_t size)
{
    const auto *bytes = static_cast<const uint8_t *>(data);
    size_t written = 0;
    while (written < size)
    {
        const ssize_t rc = send(fd, bytes + written, size - written, MSG_NOSIGNAL);
        if (rc <= 0)
        {
            return false;
        }
        written += static_cast<size_t>(rc);
    }
    return true;
}

std::string escapeJson(const std::string &input)
{
    std::ostringstream escaped;
    for (const unsigned char ch : input)
    {
        switch (ch)
        {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (ch < 0x20)
            {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
            }
            else
            {
                escaped << static_cast<char>(ch);
            }
            break;
        }
    }
    return escaped.str();
}

void deserializeEngine(const std::string &model_path,
                       nvinfer1::IRuntime *&runtime,
                       nvinfer1::ICudaEngine *&engine,
                       const char *engine_name)
{
    if (model_path.empty())
    {
        throw std::runtime_error(std::string(engine_name) + " model path is empty");
    }

    std::ifstream engine_file(model_path, std::ios::binary | std::ios::ate);
    if (!engine_file)
    {
        throw std::runtime_error(std::string("Failed to open ") + engine_name + " engine: " + model_path);
    }

    const std::streamsize engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> serialized_engine(static_cast<size_t>(engine_size));
    if (!engine_file.read(serialized_engine.data(), engine_size))
    {
        throw std::runtime_error(std::string("Failed to read ") + engine_name + " engine: " + model_path);
    }

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr)
    {
        throw std::runtime_error(std::string("Failed to create TensorRT runtime for ") + engine_name);
    }

    engine = runtime->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size());
    if (engine == nullptr)
    {
        throw std::runtime_error(std::string("Failed to deserialize ") + engine_name + " engine: " + model_path);
    }

    LOG_INFO("{} engine loaded from {} ({:.2f} MB)",
             engine_name,
             model_path,
             static_cast<double>(serialized_engine.size()) / (1024.0 * 1024.0));
}

class ThreadJoiner
{
public:
    explicit ThreadJoiner(std::thread &thread)
        : thread_(thread)
    {
    }

    ~ThreadJoiner()
    {
        if (thread_.joinable())
        {
            thread_.join();
        }
    }

private:
    std::thread &thread_;
};

class ZeroCopyFrameBuffer
{
public:
    ZeroCopyFrameBuffer(int width, int height)
        : width_(width),
          height_(height),
          bytes_(static_cast<size_t>(width) * static_cast<size_t>(height) * 3U)
    {
        auto status = cudaHostAlloc(&host_ptr_, bytes_, cudaHostAllocMapped);
        if (status != cudaSuccess)
        {
            throw std::runtime_error("cudaHostAlloc failed for zero-copy frame buffer");
        }

        status = cudaHostGetDevicePointer(&device_ptr_, host_ptr_, 0);
        if (status != cudaSuccess)
        {
            cudaFreeHost(host_ptr_);
            host_ptr_ = nullptr;
            throw std::runtime_error("cudaHostGetDevicePointer failed for zero-copy frame buffer");
        }
    }

    ~ZeroCopyFrameBuffer()
    {
        if (host_ptr_ != nullptr)
        {
            cudaFreeHost(host_ptr_);
        }
    }

    ZeroCopyFrameBuffer(const ZeroCopyFrameBuffer &) = delete;
    ZeroCopyFrameBuffer &operator=(const ZeroCopyFrameBuffer &) = delete;

    uint8_t *hostData()
    {
        return static_cast<uint8_t *>(host_ptr_);
    }

    void *deviceData()
    {
        return device_ptr_;
    }

    size_t bytes() const
    {
        return bytes_;
    }

    cv::Mat asMat()
    {
        return cv::Mat(height_, width_, CV_8UC3, host_ptr_);
    }

private:
    int width_;
    int height_;
    size_t bytes_;
    void *host_ptr_ = nullptr;
    void *device_ptr_ = nullptr;
};

class LatestJpegFrame
{
public:
    explicit LatestJpegFrame(int jpeg_quality)
        : encode_params_{cv::IMWRITE_JPEG_QUALITY, jpeg_quality}
    {
    }

    void publish(const cv::Mat &frame)
    {
        std::vector<uint8_t> encoded;
        if (!cv::imencode(".jpg", frame, encoded, encode_params_))
        {
            LOG_WARN("JPEG encode failed for browser stream");
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        jpeg_ = std::move(encoded);
        ++sequence_;
        condition_.notify_all();
    }

    bool waitForNext(std::vector<uint8_t> &jpeg, uint64_t &last_sequence)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&]() {
            return !is_run.load() || sequence_ != last_sequence;
        });

        if (!is_run.load() && sequence_ == last_sequence)
        {
            return false;
        }

        jpeg = jpeg_;
        last_sequence = sequence_;
        return !jpeg.empty();
    }

    void notifyStopped()
    {
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<uint8_t> jpeg_;
    std::vector<int> encode_params_;
    uint64_t sequence_ = 0;
};

struct FaceOverlay
{
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    float detection_score = 0.0f;
    float similarity = 0.0f;
    bool known = false;
    std::string label = "UNKNOWN";
};

class LatestInferenceResult
{
public:
    void publish(std::vector<FaceOverlay> faces,
                 int frame_width,
                 int frame_height,
                 double latency_ms,
                 std::string ai_mode,
                 uint64_t source_sequence)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        faces_ = std::move(faces);
        frame_width_ = frame_width;
        frame_height_ = frame_height;
        latency_ms_ = latency_ms;
        ai_mode_ = std::move(ai_mode);
        source_sequence_ = source_sequence;
        ++sequence_;
    }

    std::string toJson() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream json;
        json << std::fixed << std::setprecision(3);
        json << "{"
             << "\"sequence\":" << sequence_ << ","
             << "\"source_sequence\":" << source_sequence_ << ","
             << "\"frame_width\":" << frame_width_ << ","
             << "\"frame_height\":" << frame_height_ << ","
             << "\"latency_ms\":" << latency_ms_ << ","
             << "\"ai_mode\":\"" << escapeJson(ai_mode_) << "\","
             << "\"faces\":[";

        for (size_t i = 0; i < faces_.size(); ++i)
        {
            const auto &face = faces_[i];
            if (i > 0)
            {
                json << ",";
            }
            json << "{"
                 << "\"x1\":" << face.x1 << ","
                 << "\"y1\":" << face.y1 << ","
                 << "\"x2\":" << face.x2 << ","
                 << "\"y2\":" << face.y2 << ","
                 << "\"score\":" << face.detection_score << ","
                 << "\"similarity\":" << face.similarity << ","
                 << "\"known\":" << (face.known ? "true" : "false") << ","
                 << "\"label\":\"" << escapeJson(face.label) << "\""
                 << "}";
        }

        json << "]}";
        return json.str();
    }

private:
    mutable std::mutex mutex_;
    std::vector<FaceOverlay> faces_;
    int frame_width_ = 0;
    int frame_height_ = 0;
    double latency_ms_ = 0.0;
    std::string ai_mode_ = "search";
    uint64_t source_sequence_ = 0;
    uint64_t sequence_ = 0;
};

class AiFrameSlot
{
public:
    bool tryBeginWrite()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::Idle)
        {
            return false;
        }
        state_ = State::Writing;
        return true;
    }

    void finishWrite(bool has_frame, DetectionType detection_type)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = has_frame ? State::Pending : State::Idle;
            if (has_frame)
            {
                ++sequence_;
                detection_type_ = detection_type;
            }
        }
        condition_.notify_one();
    }

    bool waitForPending(uint64_t &sequence, DetectionType &detection_type)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&]() {
            return !is_run.load() || state_ == State::Pending;
        });

        if (!is_run.load() && state_ != State::Pending)
        {
            return false;
        }

        state_ = State::Running;
        sequence = sequence_;
        detection_type = detection_type_;
        return true;
    }

    void finishRun()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::Idle;
        }
        condition_.notify_one();
    }

    void notifyStopped()
    {
        condition_.notify_all();
    }

private:
    enum class State
    {
        Idle,
        Writing,
        Pending,
        Running
    };

    std::mutex mutex_;
    std::condition_variable condition_;
    State state_ = State::Idle;
    uint64_t sequence_ = 0;
    DetectionType detection_type_ = DetectionType::SEARCH;
};

class HttpServer
{
public:
    HttpServer(int port,
               std::string html_page,
               LatestJpegFrame &latest_frame,
               LatestInferenceResult &latest_inference_result)
        : port_(port),
          html_page_(std::move(html_page)),
          latest_frame_(latest_frame),
          latest_inference_result_(latest_inference_result)
    {
    }

    ~HttpServer()
    {
        stop();
    }

    void start()
    {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0)
        {
            throw std::runtime_error("Failed to create HTTP server socket");
        }

        const int enable = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(static_cast<uint16_t>(port_));

        if (bind(server_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0)
        {
            close(server_fd_);
            server_fd_ = -1;
            throw std::runtime_error("Failed to bind HTTP server socket");
        }

        if (listen(server_fd_, 16) < 0)
        {
            close(server_fd_);
            server_fd_ = -1;
            throw std::runtime_error("Failed to listen on HTTP server socket");
        }

        thread_ = std::thread([this]() { acceptLoop(); });
    }

    void stop()
    {
        if (server_fd_ >= 0)
        {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }

        if (thread_.joinable())
        {
            thread_.join();
        }
    }

private:
    void acceptLoop()
    {
        while (is_run.load())
        {
            sockaddr_in client_address{};
            socklen_t client_size = sizeof(client_address);
            const int client_fd = accept(server_fd_, reinterpret_cast<sockaddr *>(&client_address), &client_size);
            if (client_fd < 0)
            {
                if (is_run.load())
                {
                    LOG_WARN("HTTP accept failed");
                }
                break;
            }

            std::thread(&HttpServer::handleClient, this, client_fd).detach();
        }
    }

    void handleClient(int client_fd)
    {
        char request_buffer[4096] = {};
        const ssize_t received = recv(client_fd, request_buffer, sizeof(request_buffer) - 1, 0);
        if (received <= 0)
        {
            close(client_fd);
            return;
        }

        const std::string request(request_buffer, static_cast<size_t>(received));
        if (request.rfind("GET /stream.mjpg", 0) == 0)
        {
            streamMjpeg(client_fd);
        }
        else if (request.rfind("GET /detections.json", 0) == 0)
        {
            serveDetections(client_fd);
        }
        else if (request.rfind("GET /healthz", 0) == 0)
        {
            static constexpr std::string_view response =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                "Content-Length: 2\r\n"
                "Connection: close\r\n\r\n"
                "OK";
            writeAll(client_fd, response.data(), response.size());
        }
        else
        {
            serveHtml(client_fd);
        }

        close(client_fd);
    }

    void serveHtml(int client_fd)
    {
        std::ostringstream response;
        response
            << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: text/html; charset=utf-8\r\n"
            << "Cache-Control: no-cache\r\n"
            << "Connection: close\r\n"
            << "Content-Length: " << html_page_.size() << "\r\n\r\n"
            << html_page_;
        const std::string payload = response.str();
        writeAll(client_fd, payload.data(), payload.size());
    }

    void serveDetections(int client_fd)
    {
        const std::string body = latest_inference_result_.toJson();
        std::ostringstream response;
        response
            << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Cache-Control: no-store\r\n"
            << "Connection: close\r\n"
            << "Content-Length: " << body.size() << "\r\n\r\n"
            << body;
        const std::string payload = response.str();
        writeAll(client_fd, payload.data(), payload.size());
    }

    void streamMjpeg(int client_fd)
    {
        static constexpr std::string_view header =
            "HTTP/1.1 200 OK\r\n"
            "Cache-Control: no-cache\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        if (!writeAll(client_fd, header.data(), header.size()))
        {
            return;
        }

        std::vector<uint8_t> jpeg;
        uint64_t last_sequence = 0;
        while (is_run.load() && latest_frame_.waitForNext(jpeg, last_sequence))
        {
            std::ostringstream chunk_header;
            chunk_header
                << "--frame\r\n"
                << "Content-Type: image/jpeg\r\n"
                << "Content-Length: " << jpeg.size() << "\r\n\r\n";
            const std::string header_text = chunk_header.str();

            if (!writeAll(client_fd, header_text.data(), header_text.size()) ||
                !writeAll(client_fd, jpeg.data(), jpeg.size()) ||
                !writeAll(client_fd, "\r\n", 2))
            {
                return;
            }
        }
    }

    int port_;
    int server_fd_ = -1;
    std::string html_page_;
    LatestJpegFrame &latest_frame_;
    LatestInferenceResult &latest_inference_result_;
    std::thread thread_;
};

class GStreamerAppSinkSource
{
public:
    GStreamerAppSinkSource() = default;

    ~GStreamerAppSinkSource()
    {
        close();
    }

    void open(const std::string &pipeline_description)
    {
        GError *error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_description.c_str(), &error);
        if (pipeline_ == nullptr)
        {
            std::string message = error != nullptr ? error->message : "unknown gst_parse_launch failure";
            if (error != nullptr)
            {
                g_error_free(error);
            }
            throw std::runtime_error("Failed to create GStreamer pipeline: " + message);
        }

        sink_ = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), "sink"));
        if (sink_ == nullptr)
        {
            close();
            throw std::runtime_error("Failed to find appsink named 'sink' in pipeline");
        }

        gst_app_sink_set_drop(sink_, true);
        gst_app_sink_set_max_buffers(sink_, 1);
        gst_app_sink_set_wait_on_eos(sink_, false);

        const auto state_change = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (state_change == GST_STATE_CHANGE_FAILURE)
        {
            close();
            throw std::runtime_error("Failed to start GStreamer pipeline");
        }
    }

    bool copyFrameTo(cv::Mat &target_frame,
                     int expected_width,
                     int expected_height,
                     bool &frame_updated)
    {
        frame_updated = false;
        if (!drainBusMessages())
        {
            return false;
        }

        GstSample *sample = gst_app_sink_try_pull_sample(sink_, GST_SECOND / 2);
        if (sample == nullptr)
        {
            if (!drainBusMessages())
            {
                return false;
            }

            if (gst_app_sink_is_eos(sink_))
            {
                LOG_INFO("Reached end of stream");
                return false;
            }
            return true;
        }

        GstCaps *caps = gst_sample_get_caps(sample);
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        GstVideoInfo video_info{};
        gst_video_info_from_caps(&video_info, caps);

        int width = 0;
        int height = 0;
        gst_structure_get_int(structure, "width", &width);
        gst_structure_get_int(structure, "height", &height);

        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstMapInfo map{};
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ))
        {
            gst_sample_unref(sample);
            LOG_WARN("Failed to map GstBuffer");
            return true;
        }

        const auto bytes_per_pixel = 3;
        const size_t source_stride = static_cast<size_t>(video_info.stride[0] > 0
            ? video_info.stride[0]
            : width * bytes_per_pixel);
        const size_t contiguous_bytes =
            static_cast<size_t>(expected_width) * static_cast<size_t>(expected_height) * bytes_per_pixel;

        if (target_frame.empty() ||
            target_frame.cols != expected_width ||
            target_frame.rows != expected_height ||
            target_frame.type() != CV_8UC3)
        {
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            throw std::runtime_error("Target frame buffer does not match expected BGR size");
        }

        if (width == expected_width && height == expected_height && source_stride == static_cast<size_t>(expected_width * bytes_per_pixel) && map.size >= contiguous_bytes)
        {
            std::memcpy(target_frame.data, map.data, contiguous_bytes);
        }
        else
        {
            cv::Mat source(height, width, CV_8UC3, map.data, source_stride);
            cv::resize(source, target_frame, cv::Size(expected_width, expected_height));
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        frame_updated = true;
        return true;
    }

    void close()
    {
        if (pipeline_ != nullptr)
        {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        if (sink_ != nullptr)
        {
            gst_object_unref(sink_);
            sink_ = nullptr;
        }
        if (pipeline_ != nullptr)
        {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

private:
    bool drainBusMessages()
    {
        if (pipeline_ == nullptr)
        {
            return false;
        }

        GstBus *bus = gst_element_get_bus(pipeline_);
        if (bus == nullptr)
        {
            return true;
        }

        bool keep_running = true;
        while (true)
        {
            GstMessage *message = gst_bus_pop_filtered(
                bus,
                static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING));
            if (message == nullptr)
            {
                break;
            }

            switch (GST_MESSAGE_TYPE(message))
            {
            case GST_MESSAGE_ERROR:
            {
                GError *error = nullptr;
                gchar *debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                LOG_ERROR("GStreamer error: {}", error != nullptr ? error->message : "unknown error");
                if (debug != nullptr && debug[0] != '\0')
                {
                    LOG_ERROR("GStreamer debug: {}", debug);
                }
                if (error != nullptr)
                {
                    g_error_free(error);
                }
                if (debug != nullptr)
                {
                    g_free(debug);
                }
                keep_running = false;
                break;
            }
            case GST_MESSAGE_WARNING:
            {
                GError *warning = nullptr;
                gchar *debug = nullptr;
                gst_message_parse_warning(message, &warning, &debug);
                LOG_WARN("GStreamer warning: {}", warning != nullptr ? warning->message : "unknown warning");
                if (debug != nullptr && debug[0] != '\0')
                {
                    LOG_WARN("GStreamer debug: {}", debug);
                }
                if (warning != nullptr)
                {
                    g_error_free(warning);
                }
                if (debug != nullptr)
                {
                    g_free(debug);
                }
                break;
            }
            case GST_MESSAGE_EOS:
                LOG_INFO("GStreamer bus reported EOS");
                keep_running = false;
                break;
            default:
                break;
            }

            gst_message_unref(message);
            if (!keep_running)
            {
                break;
            }
        }

        gst_object_unref(bus);
        return keep_running;
    }

    GstElement *pipeline_ = nullptr;
    GstAppSink *sink_ = nullptr;
};

std::string buildCameraPipeline(const json &config, int width, int height)
{
    const int fps = kFixedCaptureFps;
    const int sensor_id = config["camera"].value("sensor_id", 0);

    if (config["camera"].contains("pipeline"))
    {
        LOG_WARN("Ignoring camera.pipeline override to keep capture fixed at {}x{}@{}",
                 width,
                 height,
                 fps);
    }

    std::ostringstream pipeline;
    pipeline
        << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
        << "video/x-raw(memory:NVMM), width=" << width
        << ", height=" << height
        << ", framerate=" << fps << "/1 ! "
        << "nvvidconv flip-method=" << kCameraFlipMethod180 << " ! "
        << "video/x-raw, format=BGRx ! "
        << "videoconvert ! video/x-raw, format=BGR ! "
        << "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        << "appsink name=sink drop=true max-buffers=1 sync=false";
    return pipeline.str();
}

std::string buildVideoPipeline(const json &config, int width, int height)
{
    const std::string video_path = config["camera"]["test_video_path"];
    if (!std::filesystem::exists(video_path))
    {
        throw std::runtime_error("Video file does not exist: " + video_path);
    }

    const int fps = kFixedCaptureFps;
    std::string extension = std::filesystem::path(video_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::ostringstream pipeline;

    if (extension == ".webm")
    {
        const std::string vp9_decoder = chooseGStreamerElement({"avdec_vp9", "vp9dec"});
        if (vp9_decoder.empty())
        {
            throw std::runtime_error(
                "No software VP9 decoder found for WebM. Install gstreamer1.0-libav "
                "(avdec_vp9) or a libvpx VP9 decoder (vp9dec).");
        }

        LOG_INFO("Using software VP9 decoder for WebM: {}", vp9_decoder);

        // Avoid decodebin on Jetson WebM/VP9: it can still pick nvv4l2decoder,
        // which reports ParseUncompressedVP9 on some files.
        pipeline
            << "filesrc location=\"" << escapeGStreamerString(video_path) << "\" ! "
            << "matroskademux name=demux "
            << "demux.video_0 ! ";
        if (hasGStreamerElement("vp9parse"))
        {
            pipeline << "vp9parse ! ";
        }
        pipeline
            << vp9_decoder << " ! "
            << "videoconvert ! "
            << "videoscale ! "
            << "videorate ! "
            << "video/x-raw, format=BGR, width=" << width
            << ", height=" << height
            << ", framerate=" << fps << "/1, pixel-aspect-ratio=1/1 ! "
            << "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
            << "appsink name=sink drop=true max-buffers=1 sync=false";
        return pipeline.str();
    }

    if (extension == ".mkv")
    {
        pipeline
            << "filesrc location=\"" << escapeGStreamerString(video_path) << "\" ! "
            << "matroskademux name=demux "
            << "demux.video_0 ! decodebin ! "
            << "nvvidconv ! video/x-raw, format=BGRx ! "
            << "videoconvert ! "
            << "videoscale ! "
            << "videorate ! "
            << "video/x-raw, format=BGR, width=" << width
            << ", height=" << height
            << ", framerate=" << fps << "/1, pixel-aspect-ratio=1/1 ! "
            << "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
            << "appsink name=sink drop=true max-buffers=1 sync=false";
        return pipeline.str();
    }

    if (extension == ".mp4" || extension == ".mov")
    {
        pipeline
            << "filesrc location=\"" << escapeGStreamerString(video_path) << "\" ! "
            << "qtdemux name=demux "
            << "demux.video_0 ! decodebin ! "
            << "nvvidconv ! video/x-raw, format=BGRx ! "
            << "videoconvert ! "
            << "videoscale ! "
            << "videorate ! "
            << "video/x-raw, format=BGR, width=" << width
            << ", height=" << height
            << ", framerate=" << fps << "/1, pixel-aspect-ratio=1/1 ! "
            << "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
            << "appsink name=sink drop=true max-buffers=1 sync=false";
        return pipeline.str();
    }

    pipeline
        << "filesrc location=\"" << escapeGStreamerString(video_path) << "\" ! "
        << "decodebin name=decoder "
        << "decoder. ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        << "nvvidconv ! video/x-raw, format=BGRx ! "
        << "videoconvert ! "
        << "videoscale ! "
        << "videorate ! "
        << "video/x-raw, format=BGR, width=" << width
        << ", height=" << height
        << ", framerate=" << fps << "/1, pixel-aspect-ratio=1/1 ! "
        << "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        << "appsink name=sink drop=true max-buffers=1 sync=false";
    return pipeline.str();
}

std::vector<FaceOverlay> buildFaceOverlays(const std::vector<DetectionBox> &detections,
                                           const std::vector<IdentificationMatch> &matches,
                                           int frame_width,
                                           int frame_height)
{
    std::vector<FaceOverlay> overlays;
    overlays.reserve(detections.size());

    for (const auto &detection : detections)
    {
        FaceOverlay overlay;
        overlay.x1 = std::clamp(detection.x1, 0, std::max(0, frame_width - 1));
        overlay.y1 = std::clamp(detection.y1, 0, std::max(0, frame_height - 1));
        overlay.x2 = std::clamp(detection.x2, 0, std::max(0, frame_width - 1));
        overlay.y2 = std::clamp(detection.y2, 0, std::max(0, frame_height - 1));
        overlay.detection_score = detection.score;
        overlay.known = false;
        overlay.label = "UNKNOWN";
        overlays.push_back(std::move(overlay));
    }

    for (const auto &match : matches)
    {
        if (match.face_index < 0 || match.face_index >= static_cast<int>(overlays.size()))
        {
            continue;
        }

        auto &overlay = overlays[static_cast<size_t>(match.face_index)];
        overlay.known = true;
        overlay.label = match.label;
        overlay.similarity = match.average_similarity;
    }

    return overlays;
}

std::string renderHtmlTemplate(const std::filesystem::path &template_path,
                               const std::string &mode,
                               const std::string &source_detail,
                               int width,
                               int height)
{
    std::string html = readTextFile(template_path);
    html = replaceAll(html, "{{STREAM_URL}}", "/stream.mjpg");
    html = replaceAll(html, "{{SOURCE_MODE}}", escapeHtml(mode));
    html = replaceAll(html, "{{SOURCE_DETAIL}}", escapeHtml(source_detail));
    html = replaceAll(html, "{{FRAME_SIZE}}", escapeHtml(std::to_string(width) + "x" + std::to_string(height)));
    html = replaceAll(html, "{{STATUS_TEXT}}", "LIVE PREVIEW");
    return html;
}

void printUsage(const char *program_name)
{
    LOG_INFO("Usage: {} [cam|video] [face_record]", program_name);
    LOG_INFO("Usage: {} cam face_record", program_name);
    LOG_INFO("Usage: {} video", program_name);
}

std::filesystem::path resolveFaceRecordDir(const std::filesystem::path &project_root)
{
    return project_root / "build" / "face_record";
}

int saveDetectedFaceCrops(const cv::Mat &frame,
                          const std::vector<DetectionBox> &detections,
                          const std::filesystem::path &face_record_dir,
                          uint64_t source_sequence,
                          int &face_record_index)
{
    if (frame.empty())
    {
        return 0;
    }

    int saved_count = 0;
    for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index)
    {
        const auto &detection = detections[detection_index];
        const int x1 = std::clamp(std::min(detection.x1, detection.x2), 0, std::max(0, frame.cols - 1));
        const int y1 = std::clamp(std::min(detection.y1, detection.y2), 0, std::max(0, frame.rows - 1));
        const int x2 = std::clamp(std::max(detection.x1, detection.x2), 0, frame.cols);
        const int y2 = std::clamp(std::max(detection.y1, detection.y2), 0, frame.rows);
        const int width = x2 - x1;
        const int height = y2 - y1;
        if (width <= 1 || height <= 1)
        {
            continue;
        }

        std::ostringstream filename;
        filename << "face_" << std::setw(6) << std::setfill('0') << face_record_index++
                 << "_seq_" << source_sequence
                 << "_det_" << detection_index
                 << ".jpg";
        if (cv::imwrite((face_record_dir / filename.str()).string(),
                        frame(cv::Rect(x1, y1, width, height)).clone()))
        {
            ++saved_count;
        }
    }

    return saved_count;
}

} // namespace

int main(int argc, char **argv)
{
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    if (argc < 2)
    {
        printUsage(argv[0]);
        return -1;
    }

    bool use_camera_source = false;
    bool use_video_source = false;
    bool face_record_mode = false;
    for (int arg_index = 1; arg_index < argc; ++arg_index)
    {
        const std::string_view arg(argv[arg_index]);
        if (arg == "cam")
        {
            use_camera_source = true;
        }
        else if (arg == "video")
        {
            use_video_source = true;
        }
        else if (arg == "face_record")
        {
            face_record_mode = true;
        }
        else
        {
            LOG_ERROR("Unknown argument: {}", arg);
            printUsage(argv[0]);
            return -1;
        }
    }

    if (use_camera_source && use_video_source)
    {
        LOG_ERROR("Use only one source parameter: 'cam' or 'video'");
        return -1;
    }

    if (!use_camera_source && !use_video_source)
    {
        use_video_source = true;
    }

    const auto project_root = resolveProjectRoot();
    const auto config_path = project_root / "configs" / "calib.json";
    const auto template_path = project_root / "templates" / "code.html";

    json data_config;
    try
    {
        loadConfig(config_path.string(), data_config);
    }
    catch (const std::exception &error)
    {
        LOG_ERROR("Failed to load config {}: {}", config_path.string(), error.what());
        return -1;
    }

    spdlog::set_level(spdlog::level::from_str(data_config["general"].value("log_level", "info")));

    const int configured_camera_width = data_config["camera"].value("width", kFixedCaptureWidth);
    const int configured_camera_height = data_config["camera"].value("height", kFixedCaptureHeight);
    const int configured_camera_fps = data_config["camera"].value("fps", kFixedCaptureFps);
    if (configured_camera_width != kFixedCaptureWidth ||
        configured_camera_height != kFixedCaptureHeight ||
        configured_camera_fps != kFixedCaptureFps)
    {
        LOG_WARN("Camera config {}x{}@{} overridden to fixed {}x{}@{}",
                 configured_camera_width,
                 configured_camera_height,
                 configured_camera_fps,
                 kFixedCaptureWidth,
                 kFixedCaptureHeight,
                 kFixedCaptureFps);
    }

    data_config["camera"]["width"] = kFixedCaptureWidth;
    data_config["camera"]["height"] = kFixedCaptureHeight;
    data_config["camera"]["fps"] = kFixedCaptureFps;

    const int camera_width = kFixedCaptureWidth;
    const int camera_height = kFixedCaptureHeight;
    const int web_port = data_config["general"].value("web_port", 8080);
    const int jpeg_quality = data_config["general"].value("stream_jpeg_quality", 80);
    const std::vector<int> detection_batch_sizes =
        data_config["detection"]["batch_sizes"].get<std::vector<int>>();
    const std::vector<int32_t> strides =
        data_config["detection"]["strides"].get<std::vector<int32_t>>();
    const std::vector<int> identification_batch_sizes =
        data_config["identification"]["batch_sizes"].get<std::vector<int>>();
    const int32_t detection_input_size = data_config["detection"]["input_size"];
    const int32_t detection_top_k = data_config["detection"]["detection_top_k"];
    const float detection_confidence_threshold = data_config["detection"]["detection_conf_threshold"];
    const float detection_iou_threshold = data_config["detection"]["detection_nms_threshold"];
    const int32_t min_box_length = data_config["identification"]["identifier_threshold_min_box_length"];
    const int32_t identification_input_size = data_config["identification"]["input_size"];
    const float face_threshold = data_config["identification"].value("face_threshold", 0.4f);
    const int32_t num_slices_x = data_config["detection"]["num_slices_x"];
    const int32_t num_slices_y = data_config["detection"]["num_slices_y"];
    const int32_t gap_x = data_config["detection"]["gap_x"];
    const int32_t gap_y = data_config["detection"]["gap_y"];
    int track_window_half_width = data_config["detection"].value("track_window_half_width", detection_input_size / 2);
    int track_window_half_height = data_config["detection"].value("track_window_half_height", detection_input_size / 2);
    double time_interval_face_record = data_config["camera"].value("time_interval_face_record", 1.0);
    if (track_window_half_width <= 0)
    {
        track_window_half_width = detection_input_size / 2;
    }
    if (track_window_half_height <= 0)
    {
        track_window_half_height = detection_input_size / 2;
    }
    if (time_interval_face_record <= 0.0)
    {
        time_interval_face_record = 1.0;
    }
    const std::string selected_faces_path = data_config["identification"].value(
        "selected_faces_path",
        (project_root / "selected_faces").string());

    nvinfer1::IRuntime *detection_runtime = nullptr;
    nvinfer1::ICudaEngine *detection_engine = nullptr;
    nvinfer1::IRuntime *identification_runtime = nullptr;
    nvinfer1::ICudaEngine *identification_engine = nullptr;
    std::unique_ptr<DetectionModelInferenceHelper> detection_helper;
    std::unique_ptr<IdentificationModelInferenceHelper> identification_helper;

    auto cleanup_tensorrt = [&]() {
        identification_helper.reset();
        detection_helper.reset();
        delete identification_engine;
        identification_engine = nullptr;
        delete detection_engine;
        detection_engine = nullptr;
        delete detection_runtime;
        detection_runtime = nullptr;
        delete identification_runtime;
        identification_runtime = nullptr;
    };

    try
    {
        deserializeEngine(data_config["detection"]["model_path"],
                          detection_runtime,
                          detection_engine,
                          "Detection");

        detection_helper = std::make_unique<DetectionModelInferenceHelper>(
            detection_engine,
            detection_batch_sizes,
            detection_input_size,
            detection_input_size,
            strides,
            detection_top_k,
            detection_confidence_threshold,
            detection_iou_threshold,
            camera_height,
            camera_width,
            num_slices_x,
            num_slices_y,
            gap_x,
            gap_y,
            min_box_length);

        if (!face_record_mode)
        {
            deserializeEngine(data_config["identification"]["model_path"],
                              identification_runtime,
                              identification_engine,
                              "Identification");
            identification_helper = std::make_unique<IdentificationModelInferenceHelper>(
                identification_engine,
                identification_batch_sizes[2],
                identification_input_size,
                identification_input_size,
                selected_faces_path,
                detection_helper->stream());
        }
    }
    catch (const std::exception &error)
    {
        LOG_ERROR("Failed to initialize AI pipeline: {}", error.what());
        cleanup_tensorrt();
        return -1;
    }

    gst_init(nullptr, nullptr);

    std::string pipeline;
    try
    {
        pipeline = use_camera_source
            ? buildCameraPipeline(data_config, camera_width, camera_height)
            : buildVideoPipeline(data_config, camera_width, camera_height);
    }
    catch (const std::exception &error)
    {
        LOG_ERROR("Failed to build GStreamer pipeline: {}", error.what());
        cleanup_tensorrt();
        gst_deinit();
        return -1;
    }

    const std::string source_detail = use_camera_source
        ? ("CAMERA SENSOR " + std::to_string(data_config["camera"].value("sensor_id", 0)))
        : std::filesystem::path(data_config["camera"]["test_video_path"].get<std::string>()).filename().string();
    std::filesystem::path face_record_dir;
    if (face_record_mode)
    {
        face_record_dir = resolveFaceRecordDir(project_root);
        std::filesystem::create_directories(face_record_dir);
    }

    LOG_INFO("Input mode: {}", use_camera_source ? "cam" : "video");
    LOG_INFO("AI pipeline enabled; display stream will keep running independently of inference");
    if (face_record_mode)
    {
        LOG_INFO("Face record mode enabled. Saving detected faces to {} every {:.2f}s",
                 face_record_dir.string(),
                 time_interval_face_record);
    }
    else
    {
        LOG_INFO("Identification DB ready on GPU with {} faces from {}",
                 identification_helper->dbCount(),
                 selected_faces_path);
    }
    LOG_INFO("Opening GStreamer source: {}", pipeline);

    if (detection_input_size > camera_width || detection_input_size > camera_height)
    {
        LOG_ERROR("Detection input size {} exceeds camera frame {}x{}",
                  detection_input_size,
                  camera_width,
                  camera_height);
        cleanup_tensorrt();
        gst_deinit();
        return -1;
    }

    ZeroCopyFrameBuffer display_frame_buffer(camera_width, camera_height);
    cv::Mat display_frame = display_frame_buffer.asMat();
    cv::Mat ai_search_input_frame(
        camera_height,
        camera_width,
        CV_8UC3,
        detection_helper->hostInputBuffer(DetectionType::SEARCH));
    cv::Mat ai_track_input_frame(
        detection_input_size,
        detection_input_size,
        CV_8UC3,
        detection_helper->hostInputBuffer(DetectionType::TRACK));

    const cv::Rect track_roi = makeCenteredRoiFromHalfExtents(
        camera_width,
        camera_height,
        track_window_half_width,
        track_window_half_height);
    const int frame_center_x = camera_width / 2;
    const int frame_center_y = camera_height / 2;
    DetectionModeController detection_mode_controller(track_roi);
    LOG_INFO("Track ROI centered at frame with half-window {}x{}: x={}, y={}, w={}, h={}",
             track_window_half_width,
             track_window_half_height,
             track_roi.x,
             track_roi.y,
             track_roi.width,
             track_roi.height);

    LatestJpegFrame latest_jpeg_frame(jpeg_quality);
    LatestInferenceResult latest_inference_result;
    AiFrameSlot ai_frame_slot;

    const std::string html_page = renderHtmlTemplate(
        template_path,
        use_camera_source ? "CAM" : "VIDEO",
        source_detail,
        camera_width,
        camera_height);

    HttpServer server(web_port, html_page, latest_jpeg_frame, latest_inference_result);
    try
    {
        server.start();
    }
    catch (const std::exception &error)
    {
        LOG_ERROR("Failed to start HTTP server: {}", error.what());
        cleanup_tensorrt();
        gst_deinit();
        return -1;
    }

    const std::string ip_address = guessLocalIpAddress();
    LOG_INFO("Browser preview: http://127.0.0.1:{}/", web_port);
    LOG_INFO("Browser preview: http://{}:{}/", ip_address, web_port);
    LOG_INFO("MJPEG stream: http://{}:{}/stream.mjpg", ip_address, web_port);
    LOG_INFO("Display zero-copy host ptr: {}", static_cast<void *>(display_frame_buffer.hostData()));
    LOG_INFO("Display zero-copy device ptr: {}", display_frame_buffer.deviceData());
    LOG_INFO("AI search buffer host ptr: {}", static_cast<void *>(ai_search_input_frame.data));
    LOG_INFO("AI track buffer host ptr: {}", static_cast<void *>(ai_track_input_frame.data));

    GStreamerAppSinkSource source;
    try
    {
        source.open(pipeline);
    }
    catch (const std::exception &error)
    {
        LOG_ERROR("Failed to open source pipeline: {}", error.what());
        is_run.store(false);
        latest_jpeg_frame.notifyStopped();
        cleanup_tensorrt();
        gst_deinit();
        return -1;
    }

    int face_record_index = 0;
    const auto face_record_interval =
        std::chrono::duration<double>(time_interval_face_record);
    auto last_face_record_time =
        std::chrono::steady_clock::now() -
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(face_record_interval);

    std::thread ai_thread([&]() {
        int reported_frames = 0;
        double elapsed_sum_ms = 0.0;
        double elapsed_min_ms = std::numeric_limits<double>::max();
        double elapsed_max_ms = 0.0;

        while (is_run.load())
        {
            uint64_t source_sequence = 0;
            DetectionType detection_type = DetectionType::SEARCH;
            if (!ai_frame_slot.waitForPending(source_sequence, detection_type))
            {
                break;
            }

            try
            {
                const auto start_time = std::chrono::high_resolution_clock::now();
                detection_helper->infer(
                    detection_type == DetectionType::SEARCH ? ai_search_input_frame.data : ai_track_input_frame.data,
                    detection_type);
                auto detections = detection_helper->getLastDetections(detection_type);
                if (detection_type == DetectionType::TRACK)
                {
                    detections = mapTrackDetectionsToFrame(
                        detections,
                        track_roi,
                        detection_input_size,
                        detection_input_size);
                }
                std::vector<IdentificationMatch> matches;
                if (face_record_mode)
                {
                    const auto now = std::chrono::steady_clock::now();
                    if (!detections.empty() &&
                        now - last_face_record_time >=
                            std::chrono::duration_cast<std::chrono::steady_clock::duration>(face_record_interval))
                    {
                        const int saved_count =
                            saveDetectedFaceCrops(ai_search_input_frame,
                                                  detections,
                                                  face_record_dir,
                                                  source_sequence,
                                                  face_record_index);
                        if (saved_count > 0)
                        {
                            last_face_record_time = now;
                            LOG_INFO("Recorded {} detected faces (seq={})", saved_count, source_sequence);
                        }
                    }
                }
                else if (!detections.empty())
                {
                    const auto warped_faces =
                        detection_helper->getDeviceWarpedFacesForIdentification(detection_type);
                    matches = identification_helper->matchWarpedFaces(
                        warped_faces.device_faces,
                        warped_faces.batch_counts,
                        warped_faces.batch_count,
                        detection_top_k,
                        face_threshold);
                }
                const auto end_time = std::chrono::high_resolution_clock::now();

                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end_time - start_time).count();
                detection_mode_controller.updateFromDetections(detections, frame_center_x, frame_center_y);
                latest_inference_result.publish(
                    buildFaceOverlays(detections, matches, camera_width, camera_height),
                    camera_width,
                    camera_height,
                    elapsed_ms,
                    detectionTypeName(detection_type),
                    source_sequence);

                elapsed_sum_ms += elapsed_ms;
                elapsed_min_ms = std::min(elapsed_min_ms, elapsed_ms);
                elapsed_max_ms = std::max(elapsed_max_ms, elapsed_ms);
                ++reported_frames;
                if (reported_frames % 30 == 0)
                {
                    LOG_INFO("AI latency (last 30): min={:.2f} mean={:.2f} max={:.2f} ms",
                             elapsed_min_ms,
                             elapsed_sum_ms / 30.0,
                             elapsed_max_ms);
                    elapsed_sum_ms = 0.0;
                    elapsed_min_ms = std::numeric_limits<double>::max();
                    elapsed_max_ms = 0.0;
                }
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("AI worker failed: {}", error.what());
                is_run.store(false);
            }

            ai_frame_slot.finishRun();
        }
    });
    ThreadJoiner ai_joiner(ai_thread);

    bool first_frame_logged = false;
    while (is_run.load())
    {
        bool frame_updated = false;
        const bool write_ai_frame = ai_frame_slot.tryBeginWrite();
        if (use_camera_source && !write_ai_frame)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        const DetectionType requested_detection_type = detection_mode_controller.requestedMode();
        cv::Mat &target_frame = write_ai_frame ? ai_search_input_frame : display_frame;
        const bool keep_running = source.copyFrameTo(
            target_frame,
            camera_width,
            camera_height,
            frame_updated);
        if (write_ai_frame)
        {
            if (frame_updated && requested_detection_type == DetectionType::TRACK)
            {
                const cv::Mat track_source = ai_search_input_frame(track_roi);
                if (track_source.cols == ai_track_input_frame.cols &&
                    track_source.rows == ai_track_input_frame.rows)
                {
                    track_source.copyTo(ai_track_input_frame);
                }
                else
                {
                    cv::resize(track_source,
                               ai_track_input_frame,
                               ai_track_input_frame.size(),
                               0.0,
                               0.0,
                               cv::INTER_LINEAR);
                }
            }
            ai_frame_slot.finishWrite(frame_updated, requested_detection_type);
        }

        if (!keep_running)
        {
            is_run.store(false);
            break;
        }

        if (!frame_updated)
        {
            continue;
        }

        // Camera mode can outpace AI significantly. Publishing preview frames that never
        // entered the AI slot makes detections/identity overlays drift onto different
        // images, which looks like matched faces never turn green. Keep camera preview
        // pinned to frames that were actually submitted to inference.
        const bool should_publish_frame =
            !target_frame.empty() &&
            (!use_camera_source || write_ai_frame);
        if (should_publish_frame)
        {
            latest_jpeg_frame.publish(target_frame);
            if (!first_frame_logged)
            {
                LOG_INFO("First frame received and published to browser");
                first_frame_logged = true;
            }
        }
    }

    source.close();
    ai_frame_slot.notifyStopped();
    if (ai_thread.joinable())
    {
        ai_thread.join();
    }
    latest_jpeg_frame.notifyStopped();
    server.stop();
    gst_deinit();
    cleanup_tensorrt();

    LOG_INFO("AI preview flow stopped");
    return 0;
}
