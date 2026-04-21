import cv2
import numpy as np
import onnxruntime as ort
import json
import argparse


def make_positions(dim: int, win: int, gap: int, num: int) -> list[int]:
    step = win - gap
    span = (num - 1) * step + win
    pad  = (dim - span) // 2
    return [pad + i * step for i in range(num)]


def get_slices(img_w, img_h, model_w, model_h, gap_x, gap_y, num_x, num_y):
    xs = make_positions(img_w, model_w, gap_x, num_x)
    ys = make_positions(img_h, model_h, gap_y, num_y)
    slices = []
    for y in ys:
        for x in xs:
            slices.append((x, y, x + model_w, y + model_h))
    return slices


def preprocess(img_bgr: np.ndarray, slices: list, model_h: int, model_w: int, swap_rb: bool = True) -> np.ndarray:
    batch = np.empty((len(slices), 3, model_h, model_w), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(slices):
        crop = img_bgr[y1:y2, x1:x2]
        if swap_rb:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # RGB HWC
        chw = crop.transpose(2, 0, 1)                      # CHW
        batch[i] = (chw.astype(np.float32) - 127.5) / 128.0
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/user/Documents/rfr/configs/calib.json")
    parser.add_argument("--model",  default="/home/user/Documents/rfr/models/det_10g.onnx")
    parser.add_argument("--image",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cam_w    = cfg["camera"]["width"]
    cam_h    = cfg["camera"]["height"]
    model_sz = cfg["detection"]["input_size"]
    gap_x    = cfg["detection"]["gap_x"]
    gap_y    = cfg["detection"]["gap_y"]
    num_x    = cfg["detection"]["num_slices_x"]
    num_y    = cfg["detection"]["num_slices_y"]
    conf_thr = cfg["detection"]["conf_threshold"]
    image_path = args.image or cfg["detection"]["test_image_path"]

    slices = get_slices(cam_w, cam_h, model_sz, model_sz, gap_x, gap_y, num_x, num_y)
    print(f"Slices ({num_x}x{num_y}):")
    for i, s in enumerate(slices):
        print(f"  [{i}] {s}")

    img = cv2.imread(image_path)
    assert img is not None, f"Cannot load image: {image_path}"
    img = cv2.resize(img, (cam_w, cam_h))

    batch = preprocess(img, slices, model_sz, model_sz)
    print(f"Batch shape: {batch.shape}  dtype: {batch.dtype}")
    
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"Input: {input_name}  shape: {sess.get_inputs()[0].shape}")
    print(f"Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    for b, (x1, y1, x2, y2) in enumerate(slices):
        single  = batch[b:b+1]
        outputs = sess.run(None, {input_name: single})

        # 9 outputs: scores_s8, scores_s16, scores_s32, bboxes_s8, ..., lm_s8, ...
        scores_b = np.concatenate([outputs[0], outputs[1], outputs[2]], axis=0).squeeze(-1)  # [anchors]

        max_score = scores_b.max()
        det_count = (scores_b > conf_thr).sum()
        print(f"Batch {b} | max_score: {max_score:.4f} | dets>{conf_thr}: {det_count}")

    for i, (x1, y1, x2, y2) in enumerate(slices):
        crop = img[y1:y2, x1:x2].copy()
        cv2.putText(crop, f"slice {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(f"slice_{i}", crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
