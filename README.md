
# Introduction
This project uses the YOLOv5 model and Deepsort algorithm to achieve object tracking and counting. The code for each part has been encapsulated and can be easily embedded into your own project. In this project, you can add parameters to specify the video or local camera that needs to be inferred; You can also specify whether the parameter will save the tracked data, which includes target category and bounding box information; You can also add parameters to track the targets you are interested in and save the corresponding tracking data. Below are some detailed introduction, hoping to get your star!


# Prepare
1 Create a virtual environment 

    conda env create -f environment.yml   
    conda activate yolov5_deepsort   

2 Install all dependencies

    pip install -r requirements.txt

3 Download weight file (optional)

    Download the yolov5 weight. I already put the yolov5s.pt inside. If you need other models, please go to official site of yolov5. and place the downlaoded .pt file under ./weights/yolov5s.pt. And I also aready downloaded the deepsort weights. You can also download it from here, and place ckpt.t7 file under ./deep_sort/deep_sort/deep/checkpoint/ckpt.t7


# Run

    cd yolov5_deepsort

If you want to track the target in this video or camera, you can run this script by replacing the parameter after -- source with the video file path or 0 (webcam), which is a required parameter. Then you can choose to set -- save_video, which is an output video file name, but it is not necessary. If you do not set it, it will default to result.mp4 and the output video file will be saved Under the directory of ./results/video.

    python detect.py --source        #(necessary)The input video,you must input file name or 0(webcam)
                    --save_video     #(optional)Path to save the output video file.

If you need to save the information of the tracked target in the video, you can add the parameter -- save_data. There is no need to fill in any information after this parameter. If you select this parameter, it will be displayed Generate a. json file in the ./results/object , where each line represents the tracking data for the corresponding frame. The data for each frame is saved in a dictionary format, where the key represents the category information for the target and the value represents the bounding box information for the corresponding category.

    python detect.py --source        #(necessary) The input video,you can choice file or 0(webcam)
                    --save_video     #(optional) Path to save the output video file.
                    --save_data      #(optional) If you need to save the detection data, you can add this parameter, which will generate a json


If you want to set tracking for a specific category, you can set the -- category parameter, which must list the category names you want to track. You can refer to YOLOv5's official 80 categories for category names. Please note that if you add this parameter but do not add any category names afterwards, it will not track any categories, which is not recommended.

    python detect.py --source        #(necessary) The input video,you must input file name or 0(webcam)
                    --save_video     #(optional) Path to save the output video file.
                    --save_data      #(optional) If you need to save the detection data, you can add this parameter, which will generate a json
                    --category       #(necessary) You must fill in the category name at the end


# Others
## yolov5 detector：

```python
class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        from detect import parse_opt
        args = parse_opt()
        self.weights = args.weights
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):
        from detect import parse_opt
        args = parse_opt()
        im0, img = self.preprocess(im)
        pred = self.m(img, augment=False)[0]
        
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.45)
        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if args.category is not None and lbl not in args.category:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

```
Call the self.detetect method to return the image and prediction results

## deepsort tracker：

```python
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
```

Call the self.update method to update the tracking results

## create detector：

```python
from AIDetector_pytorch import Detector
det = Detector()
```

## call the detection interface：

```python
result = det.feedCap(im)
```

Among them, im is the BGR image, the returned result is the dictionary, and result ['frame '] returns the visualized image

## contact author：

> Github：https://github.com/HAOYON-666
> QQ group: 679035342

Following the GNU General Public License v3.0 protocol, indicate the source of the object detection section: https://github.com/ultralytics/yolov5/


# yolov5_deepsort
# yolov5_deepsort
# yolov5_deepsort
