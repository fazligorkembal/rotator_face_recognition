import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort


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


def supports_requested_batch(sess, requested_batch_size: int) -> bool:
    batch_dim = sess.get_inputs()[0].shape[0]
    if isinstance(batch_dim, int):
        return batch_dim == requested_batch_size
    return True


def save_raw_output(path: str, array: np.ndarray) -> None:
    np.savetxt(path, np.asarray(array, dtype=np.float32).reshape(-1), fmt="%.8f")


def split_output_batches(output_value: np.ndarray, batch_size: int) -> tuple[list[np.ndarray], np.ndarray]:
    arr = np.asarray(output_value, dtype=np.float32)
    if arr.ndim >= 2 and arr.shape[0] == batch_size:
        return [arr[batch_idx] for batch_idx in range(batch_size)], arr.reshape(-1)

    flat = arr.reshape(-1)
    if flat.size % batch_size != 0:
        raise ValueError(
            f"Cannot split output of {flat.size} values into {batch_size} equal batches"
        )

    elems_per_batch = flat.size // batch_size
    batch_views = [
        flat[batch_idx * elems_per_batch:(batch_idx + 1) * elems_per_batch]
        for batch_idx in range(batch_size)
    ]
    return batch_views, flat


def dump_outputs(build_dir: str, outputs: list[np.ndarray], output_infos, batch_size: int) -> None:
    for output_idx, (output_info, output_value) in enumerate(zip(output_infos, outputs)):
        batch_views, all_batches = split_output_batches(output_value, batch_size)
        save_raw_output(
            os.path.join(build_dir, f"onnx_raw_output{output_idx}_all_batches.txt"),
            all_batches,
        )
        print(
            f"Saved raw output {output_idx} ({output_info.name}) all_batches "
            f"shape={all_batches.shape}"
        )

        for batch_idx, batch_value in enumerate(batch_views):
            save_raw_output(
                os.path.join(build_dir, f"onnx_raw_output{output_idx}_batch{batch_idx}.txt"),
                batch_value,
            )
            print(
                f"Saved raw output {output_idx} ({output_info.name}) batch {batch_idx} "
                f"shape={np.asarray(batch_value).shape}"
            )


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
    image_path = args.image or cfg["detection"]["test_image_path"]

    slices = get_slices(cam_w, cam_h, model_sz, model_sz, gap_x, gap_y, num_x, num_y)
    img = cv2.imread(image_path)
    assert img is not None, f"Cannot load image: {image_path}"
    img = cv2.resize(img, (cam_w, cam_h))
    batch = preprocess(img, slices, model_sz, model_sz)

    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build/logresults")
    os.makedirs(build_dir, exist_ok=True)

    save_raw_output(os.path.join(build_dir, "onnx_input_all_batches.txt"), batch)
    print(f"Saved ONNX input all_batches shape={batch.shape}")
    for batch_idx in range(batch.shape[0]):
        save_raw_output(os.path.join(build_dir, f"onnx_input_batch{batch_idx}.txt"), batch[batch_idx])
        print(f"Saved ONNX input batch {batch_idx} shape={batch[batch_idx].shape}")

    print(f"Image path: {image_path}")
    print(f"Build dir: {os.path.abspath(build_dir)}")
    print(f"Batch shape: {batch.shape} dtype={batch.dtype}")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_infos = sess.get_outputs()
    model_batched = supports_requested_batch(sess, batch.shape[0])

    print("Session inputs:")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"  [{i}] {inp.name} shape={inp.shape} type={inp.type}")
    print("Session outputs:")
    for i, out in enumerate(output_infos):
        print(f"  [{i}] {out.name} shape={out.shape} type={out.type}")
    print(f"Model batch support: {model_batched}")

    if model_batched:
        outputs = sess.run(None, {input_name: batch})
        dump_outputs(build_dir, outputs, output_infos, batch.shape[0])
    else:
        per_batch_outputs = []
        for batch_idx in range(batch.shape[0]):
            per_batch_outputs.append(sess.run(None, {input_name: batch[batch_idx:batch_idx + 1]}))

        merged_outputs = []
        for output_idx in range(len(output_infos)):
            merged_outputs.append(np.concatenate([run[output_idx] for run in per_batch_outputs], axis=0))
        dump_outputs(build_dir, merged_outputs, output_infos, batch.shape[0])


if __name__ == "__main__":
    main()
