import os.path as osp
import cv2
import numpy as np
import onnxruntime
from loguru import logger
from skimage import transform as trans


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file=None, session=None):

        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()
        logger.info(f"Initialize SCRFD model from {self.model_file}")

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (
                    area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
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


class R50:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = "identification"
        self.batched = False

        self.arcface_src = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        self.arcface_src = np.expand_dims(self.arcface_src, axis=0)

        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)

        self.input_name = self.session.get_inputs()[0].name

        self.output_names = []
        for o in self.session.get_outputs():
            self.output_names.append(o.name)

        logger.info(
            f"R50 model input name: {self.input_name}, input shape: {self.session.get_inputs()[0].shape}, input type: {self.session.get_inputs()[0].type}"
        )
        logger.info(
            f"R50 model output names: {self.output_names}, output shapes: {[o.shape for o in self.session.get_outputs()]}, output types: {[o.type for o in self.session.get_outputs()]}"
        )

        logger.info(f"Initialize R50 model from {self.model_file}")

    # lmk is prediction; src is template
    def estimate_norm(self, lmk, image_size=112, mode="arcface"):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float("inf")
        if mode == "arcface":
            if image_size == 112:
                src = self.arcface_src
            else:
                src = float(image_size) / 112 * self.arcface_src

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, img, landmark, image_size=112, mode="arcface"):
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def detect(self, img, results_det, name="unknown"):
        bboxes = results_det[0]
        kpss = results_det[1]
        face_imgs = []
        for i, box in enumerate(bboxes):
            if kpss is not None:
                kps = kpss[i]
                face_img = self.norm_crop(img, kps, image_size=112, mode="arcface")
                blob = cv2.dnn.blobFromImage(
                    face_img, 1.0 / 128, (112, 112), (127.5, 127.5, 127.5), swapRB=True
                )
                cv2.imwrite(
                    f"/home/user/Documents/tensorrt_scrfd/datasets/tensorrt_scrfd/cropped_images/{name}_{i}.jpg",
                    face_img,
                )
                net_outs = self.session.run(self.output_names, {self.input_name: blob})
                feature = net_outs[0]
                feature = feature.flatten()
                face_imgs.append((face_img, feature))

        return face_imgs


if __name__ == "__main__":
    model_path = "/home/user/Documents/rfr/models/det_10g.onnx"
    #model_path_r50 = "/home/user/Documents/tensorrt_scrfd/models/r50.onnx"
    scrfd_model = SCRFD(model_file=model_path)

    #r50_model = R50(model_file=model_path_r50)

    # image_path = "/home/user/Documents/tensorrt_scrfd/datasets/tensorrt_scrfd/tbbt.jpg"
    image_path = (
        "/home/user/Documents/rfr/datasets/rfr/test1.png"
    )
    img = cv2.imread(image_path)
    results_det = scrfd_model.detect(img, thresh=0.5, input_size=(640, 640))
    #results = r50_model.detect(img, results_det, name="sheldon1")

    #image2_path = "/home/user/Documents/tensorrt_scrfd/datasets/tensorrt_scrfd/tbbt.jpg"
    #img2 = cv2.imread(image2_path)
    #results_det2 = scrfd_model.detect(img2, thresh=0.5, input_size=(640, 640))
    #results_2 = r50_model.detect(img2, results_det2, name="tbbt")

    #results_reference = results[0][1]

    #for i, (face_img, feature) in enumerate(results_2):
    #    similarity = np.dot(results_reference, feature) / (
    #        np.linalg.norm(results_reference) * np.linalg.norm(feature)
    #    )
    #    print(f"Similarity of face {i}: {similarity:.4f}")
    #    if similarity > 0.5:
   #         cv2.imshow(f"matched_face_{i}", face_img)
    cv2.imshow("input_image_1", img)
    #cv2.imshow("input_image_2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    dets, kpss = scrfd_model.detect(img, thresh=0.5, input_size=(640, 640))
    for i in range(dets.shape[0]):
        box = dets[i, :4].astype(np.int32)
        score = dets[i, 4]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{score:.2f}",
            (box[0], box[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        if kpss is not None:
            kps = kpss[i]
            for k in range(kps.shape[0]):
                kp = kps[k].astype(np.int32)
                cv2.circle(img, (kp[0], kp[1]), 2, (0, 0, 255), -1)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
