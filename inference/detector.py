import cv2
import torch
import numpy as np

from model import *
from torchvision import ops
from torch import nn, Tensor
from typing import List, Tuple, Dict, Union, Optional, Generator

import util


def chunks(lst: list, size: Optional[int] = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


class Predictor:
    def __init__(
        self,
        model: dict = None,
        weight_path: str = None,
        batch_size: Optional[str] = None,
        image_size: int = 640,
        classes: str = None,
        mean: Tuple[float, float, float] = (0., 0., 0.),
        std: Tuple[float, float, float] = (1., 1., 1.),
        anchors: List[List[Tuple[float, float]]] = None,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = 'cpu',
    ):
        super(Predictor, self).__init__()
        self.device = device
        self.anchors = anchors
        self.batch_size = batch_size
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mean = torch.tensor(mean, dtype=torch.float, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float, device=device).view(1, 3, 1, 1)
        class_list = open(classes, "r")
        self.classes = {i: name[:-1] for i, name in enumerate(class_list)}

        self.model = util.create_instance(model)
        # load_darknet_weights(self.model, weight_path)
        state_dict = torch.load(weight_path, map_location=device)['model']
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    def __call__(self, images: List[np.ndarray]) -> List[Dict[str, Tensor]]:
        samples = self.preprocess(images)
        preds = self.process(samples)
        outputs = self.postprocess(preds)

        for i in range(len(images)):
            if outputs[i]['labels'] is not None:
                ratio = max(images[i].shape[:2]) / self.image_size
                outputs[i]['boxes'] *= ratio
                outputs[i]['names'] = [self.classes[label.item()] for label in outputs[i]['labels']]

        return outputs

    def preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Args:
            images: list of images (image: H x W x C)
        Outputs:
            samples: list of processed images (sample: 416 x 416 x 3)
        '''
        samples = []
        for image in images:
            sample = self._resize(image, imsize=self.image_size)
            sample = self._pad_to_square(sample)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

            samples.append(sample)

        return samples

    def process(self, samples: List[np.ndarray]):
        '''
        Args:
            samples: list of processed images (sample: 416 x 416 x 3)
        Outputs:
            preds: list of three tensor for three scales (S1=13, S2=26, S3=52)
                . N x 3 x S1 x S1 x (5 + C)
                . N x 3 x S2 x S2 x (5 + C)
                . N x 3 x S3 x S3 x (5 + C)
            with:
                N: num of input images
                3: num of anchors for each scales,
                5: num pf predicted values for each boxes (tp, tx, ty, tw, th),
                C: num classes,
        '''
        # s1_preds, s2_preds, s3_preds = [], [], []

        for batch in chunks(samples, size=self.batch_size):
            batch = [torch.from_numpy(sample) for sample in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = batch.float().div(255.)

            with torch.no_grad():
                # print(len(self.model(batch)))  
                preds = self.model(batch)[0]

                # s1_preds += torch.split(tensor=s1, split_size_or_sections=1, dim=0)
                # s2_preds += torch.split(tensor=s2, split_size_or_sections=1, dim=0)
                # s3_preds += torch.split(tensor=s3, split_size_or_sections=1, dim=0)

        # preds = (
        #     torch.cat(s1_preds, dim=0),  # N x 3 x S1 x S1 x (5 + C)
        #     torch.cat(s2_preds, dim=0),  # N x 3 x S2 x S2 x (5 + C)
        #     torch.cat(s3_preds, dim=0),  # N x 3 x S3 x S3 x (5 + C)
        # )

        return preds    

    def postprocess(self, preds):
        nc = preds[0].shape[1] - 5  # number of classes
        xc = preds[..., 4] > self.score_threshold  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = []
        for i, x in enumerate(preds):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[i]]  # confidence (lay ra nhung tensor co: score > self.score_threshold)

            # If none remain process next image
            if not x.shape[0]:
                output.append({
                'boxes': None,
                'labels': None,
                'scores': None
            })
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > self.score_threshold).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.score_threshold]

            n = x.shape[0]  # number of boxes
            if not n:
                output.append({
                'boxes': None,
                'labels': None,
                'scores': None
            })
                continue

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torch.ops.torchvision.nms(boxes, scores, self.iou_threshold)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output.append({
                'boxes': x[i][:, :4],
                'labels': x[i][:, 5:6],
                'scores': x[i][:, 4:5]
            })
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    def _resize(self, image: np.ndarray, imsize=416) -> Tuple[np.ndarray, float]:
        ratio = imsize / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
        return image

    def _pad_to_square(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_size = max(height, width)
        image = np.pad(
            image, ((0, max_size - height), (0, max_size - width), (0, 0))
        )
        return image


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.5, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence (lay ra nhung tensor co: score > conf_thres)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


if __name__ == '__main__':
    x = cv2.imread('persons.jpg')
    print(x.shape)
    config = util.load_yaml('config.yaml')
    predictor = util.create_instance(config)
    print(predictor([x]))