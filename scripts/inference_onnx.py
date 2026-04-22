import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def nms(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def make_positions(dim: int, win: int, gap: int, num: int) -> list[int]:
    step = win - gap
    span = (num - 1) * step + win
    pad = (dim - span) // 2
    return [pad + i * step for i in range(num)]


def get_slices(img_w, img_h, model_w, model_h, gap_x, gap_y, num_x, num_y):
    xs = make_positions(img_w, model_w, gap_x, num_x)
    ys = make_positions(img_h, model_h, gap_y, num_y)
    slices = []
    for y in ys:
        for x in xs:
            slices.append((x, y, x + model_w, y + model_h))
    return slices


def preprocess(img_bgr: np.ndarray, slices: list, model_h: int, model_w: int) -> np.ndarray:
    batch = np.empty((len(slices), 3, model_h, model_w), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(slices):
        crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        batch[i] = (crop.transpose(2, 0, 1).astype(np.float32) - 127.5) / 128.0
    return batch


def make_anchor_centers(height, width, stride, num_anchors, cache):
    key = (height, width, stride)
    if key not in cache:
        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)
        if num_anchors > 1:
            centers = np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)
        if len(cache) < 100:
            cache[key] = centers
    return cache[key]


def decode_scrfd(outputs, batch_idx, model_h, model_w, conf_thr, center_cache, model_batched, raw=False):
    num_outputs = len(outputs)
    if num_outputs == 9:
        fmc, strides, num_anchors, use_kps = 3, [8, 16, 32], 2, True
    elif num_outputs == 6:
        fmc, strides, num_anchors, use_kps = 3, [8, 16, 32], 2, False
    else:
        raise ValueError(f"Unexpected SCRFD output count: {num_outputs}")

    all_scores = []
    all_bboxes = []
    all_kps = []

    for i, stride in enumerate(strides):
        if model_batched:
            scores_raw = outputs[i][batch_idx]
            bbox_raw = outputs[i + fmc][batch_idx] * stride
            kps_raw = outputs[i + fmc * 2][batch_idx] * stride if use_kps else None
        else:
            scores_raw = outputs[i]
            bbox_raw = outputs[i + fmc] * stride
            kps_raw = outputs[i + fmc * 2] * stride if use_kps else None

        scores = scores_raw.squeeze(-1)
        height = model_h // stride
        width = model_w // stride
        anchors = make_anchor_centers(height, width, stride, num_anchors, center_cache)

        if raw:
            pos = np.arange(len(scores))
        else:
            pos = np.where(scores >= conf_thr)[0]

        all_scores.append(scores[pos])
        all_bboxes.append(distance2bbox(anchors, bbox_raw)[pos])
        if use_kps:
            all_kps.append(distance2kps(anchors, kps_raw).reshape(-1, 5, 2)[pos])

    if not all_scores:
        return np.array([]), np.zeros((0, 4), dtype=np.float32), None

    scores = np.concatenate(all_scores)
    bboxes = np.concatenate(all_bboxes, axis=0)
    kps = np.concatenate(all_kps, axis=0) if use_kps else None

    if raw:
        return scores, bboxes, kps

    pre_det = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
    keep = nms(pre_det)
    return scores[keep], bboxes[keep], kps[keep] if kps is not None else None


def probe_batch_support(sess, input_name, model_h, model_w) -> bool:
    dummy = np.zeros((1, 3, model_h, model_w), dtype=np.float32)
    outputs = sess.run(None, {input_name: dummy})
    return outputs[0].ndim == 3


def run_inference(sess, input_name, batch, slices, model_h, model_w, conf_thr, model_batched, center_cache, raw=False):
    if model_batched:
        outputs = sess.run(None, {input_name: batch})
        return [
            decode_scrfd(outputs, b, model_h, model_w, conf_thr, center_cache, True, raw=raw)
            for b in range(len(slices))
        ]

    results = []
    for b in range(len(slices)):
        outputs = sess.run(None, {input_name: batch[b:b + 1]})
        results.append(
            decode_scrfd(outputs, 0, model_h, model_w, conf_thr, center_cache, False, raw=raw)
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/user/Documents/rfr/configs/calib.json")
    parser.add_argument("--model", default="/home/user/Documents/rfr/models/det_10g.onnx")
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cam_w = cfg["camera"]["width"]
    cam_h = cfg["camera"]["height"]
    model_sz = cfg["detection"]["input_size"]
    gap_x = cfg["detection"]["gap_x"]
    gap_y = cfg["detection"]["gap_y"]
    num_x = cfg["detection"]["num_slices_x"]
    num_y = cfg["detection"]["num_slices_y"]
    conf_thr = cfg["detection"]["conf_threshold"]
    image_path = args.image or cfg["detection"]["test_image_path"]

    slices = get_slices(cam_w, cam_h, model_sz, model_sz, gap_x, gap_y, num_x, num_y)
    print(f"Image path: {image_path}")
    print(f"Build dir: {os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build'))}")
    print(f"Slices ({num_x}x{num_y}):")
    for i, s in enumerate(slices):
        print(f"  [{i}] {s}")

    img = cv2.imread(image_path)
    assert img is not None, f"Cannot load image: {image_path}"
    img = cv2.resize(img, (cam_w, cam_h))
    vis = img.copy()

    batch = preprocess(img, slices, model_sz, model_sz)
    print(f"Batch shape: {batch.shape}  dtype: {batch.dtype}")

    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
    os.makedirs(build_dir, exist_ok=True)
    for i in range(len(slices)):
        np.savetxt(os.path.join(build_dir, f"onnx_input_slice{i}.txt"), batch[i].flatten(), fmt="%.8f")
        rgb = (batch[i].transpose(1, 2, 0) * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(build_dir, f"onnx_input_slice{i}.png"), bgr)
        print(
            f"Saved slice {i} inputs -> "
            f"onnx_input_slice{i}.txt, onnx_input_slice{i}.png"
        )

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    model_batched = probe_batch_support(sess, input_name, model_sz, model_sz)
    print("Session inputs:")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"  [{i}] {inp.name} shape={inp.shape} type={inp.type}")
    print("Session outputs:")
    for i, out in enumerate(sess.get_outputs()):
        print(f"  [{i}] {out.name} shape={out.shape} type={out.type}")
    print(f"Model batch support: {model_batched}")

    center_cache = {}
    results = run_inference(sess, input_name, batch, slices, model_sz, model_sz, conf_thr, model_batched, center_cache)
    raw_results = run_inference(sess, input_name, batch, slices, model_sz, model_sz, conf_thr, model_batched, center_cache, raw=True)

    for b, (scores, bboxes, kps) in enumerate(raw_results):
        np.savetxt(os.path.join(build_dir, f"onnx_output_slice{b}_scores.txt"), scores.flatten(), fmt="%.8f")
        np.savetxt(os.path.join(build_dir, f"onnx_output_slice{b}_bboxes.txt"), bboxes.flatten(), fmt="%.8f")
        if kps is not None:
            np.savetxt(os.path.join(build_dir, f"onnx_output_slice{b}_landmarks.txt"), kps.flatten(), fmt="%.8f")
        else:
            open(os.path.join(build_dir, f"onnx_output_slice{b}_landmarks.txt"), "w").close()
        print(
            f"Saved slice {b} raw outputs -> "
            f"onnx_output_slice{b}_scores.txt ({len(scores)}) | "
            f"onnx_output_slice{b}_bboxes.txt {bboxes.shape} | "
            f"onnx_output_slice{b}_landmarks.txt {kps.shape if kps is not None else None}"
        )

    slice_vis_list = []
    for b, ((x1_sl, y1_sl, _, _), (scores, bboxes, kps)) in enumerate(zip(slices, results)):
        print(
            f"Slice {b} | dets>{conf_thr}: {len(scores)}"
            + (f" | max_score: {scores.max():.4f}" if len(scores) else "")
        )

        slice_vis = img[y1_sl:y1_sl + model_sz, x1_sl:x1_sl + model_sz].copy()
        for j, (box, score) in enumerate(zip(bboxes, scores)):
            cv2.rectangle(
                slice_vis,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                slice_vis,
                f"{score:.2f}",
                (int(box[0]), max(12, int(box[1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
            if kps is not None:
                for kp in kps[j]:
                    cv2.circle(slice_vis, (int(kp[0]), int(kp[1])), 2, (255, 0, 0), -1)

            gx1 = int(box[0]) + x1_sl
            gy1 = int(box[1]) + y1_sl
            gx2 = int(box[2]) + x1_sl
            gy2 = int(box[3]) + y1_sl
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{score:.2f}",
                (gx1, max(12, gy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
            if kps is not None:
                for kp in kps[j]:
                    cv2.circle(vis, (int(kp[0]) + x1_sl, int(kp[1]) + y1_sl), 2, (255, 0, 0), -1)

        cv2.putText(slice_vis, f"slice {b}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        slice_vis_list.append(slice_vis)
        cv2.imwrite(os.path.join(build_dir, f"onnx_slice{b}_detections.png"), slice_vis)
        print(
            f"Saved slice {b} detection visual -> "
            f"onnx_slice{b}_detections.png"
        )

    for x1_sl, y1_sl, x2_sl, y2_sl in slices:
        cv2.rectangle(vis, (x1_sl, y1_sl), (x2_sl, y2_sl), (255, 255, 0), 1)

    cv2.imwrite(os.path.join(build_dir, "onnx_detections_full.png"), vis)
    print("Saved full detection visual -> onnx_detections_full.png")
    cv2.imshow("original (full)", img)
    cv2.imshow("detections (full)", vis)
    for i, slice_vis in enumerate(slice_vis_list):
        row, col = divmod(i, num_x)
        cv2.imshow(f"slice {i} (row={row} col={col})", slice_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
