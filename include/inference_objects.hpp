#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

class SCRFDResults
{
public:
    explicit SCRFDResults(size_t max_detections)
    {
        scores_ = (float *)malloc(max_detections * sizeof(float));
        bboxes_ = (float4 *)malloc(max_detections * sizeof(float4));
        landmarks_ = (float2 *)malloc(max_detections * 5 * sizeof(float2));
        img_display_ = (uint8_t *)malloc(720 * 1280 * 3 * sizeof(uint8_t));
        faces_warped_ = (float *)malloc(512 * 3 * 112 * 112 * sizeof(float));
    }

    ~SCRFDResults()
    {
        if (scores_)
        {
            free(scores_);
            scores_ = nullptr;
        }
        if (bboxes_)
        {
            free(bboxes_);
            bboxes_ = nullptr;
        }
        if (landmarks_)
        {
            free(landmarks_);
            landmarks_ = nullptr;
        }
        if (img_display_)
        {
            free(img_display_);
            img_display_ = nullptr;
        }
        if (faces_warped_)
        {
            free(faces_warped_);
            faces_warped_ = nullptr;
        }
    }

    float *scores_{nullptr};
    float4 *bboxes_{nullptr};
    float2 *landmarks_{nullptr};
    uint8_t *img_display_{nullptr};
    float *faces_warped_{nullptr};
    std::vector<int> matched_ids_;
    int32_t detected_count_{0};
    int32_t frame_number_{-1};
};