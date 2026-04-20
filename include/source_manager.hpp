extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <algorithm>
#include <spdlog/spdlog.h>

struct FrameData
{
    AVFrame *frame;
    int64_t frame_number;
};

class SourceManager
{
public:
    SourceManager(size_t width_output, size_t height_output, size_t num_threads);
    ~SourceManager();
    bool open_video(const char *filename);
    void start_decoding(int frame_skip);
    void start_multithread_decoding(int frame_skip);
    bool get_next_frame(int thread_id, uint8_t *&current_frame_data, int64_t &frame_number);
    void stop();
    size_t get_current_queue_size()
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        spdlog::info("Getting current queue size: {}", frame_queue.size());
        return frame_queue.size();
    }

    int get_frame_linesize() const
    {
        return !rgb_frames_pool.empty() && rgb_frames_pool[0] ? rgb_frames_pool[0]->linesize[0] : 0;
    }

private:
    AVFormatContext *format_ctx = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    int video_stream_index = -1;
    std::queue<FrameData> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> stop_decoding{false};
    std::thread decoding_thread;
    AVDictionary *opts = nullptr;
    size_t max_queue_size = 20;
    size_t width_output_;
    size_t height_output_;
    std::vector<AVFrame*> rgb_frames_pool;
    std::vector<SwsContext*> sws_contexts_pool;
};