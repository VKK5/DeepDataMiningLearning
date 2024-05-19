import os
import torch
import torchvision
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models import get_model, get_model_weights, list_models
from torchvision.models.detection import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from DeepDataMiningLearning.detection.modules.yolomodels import create_yolomodel, freeze_yolomodel
from DeepDataMiningLearning.detection.modeling_rpnfasterrcnn import CustomRCNN

try:
    from torchinfo import summary
except ImportError:
    print("[INFO] Couldn't find torchinfo... installing it.") # pip install -q torchinfo

# Function to get torchvision detection models
def get_torchvision_detection_models(modelname, box_score_thresh=0.9):
    weights_enum = get_model_weights(modelname)
    weights = weights_enum.DEFAULT  # Get the default weights
    preprocess = weights.transforms()
    classes = weights.meta["categories"]
    pretrained_model = get_model(modelname, box_score_thresh=box_score_thresh, weights="DEFAULT")
    return pretrained_model, preprocess, weights, classes

# Function to modify FasterRCNN header
def modify_fasterrcnnheader(model, num_classes, freeze=True):
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Function to load a trained model
def load_trained_model(modelname, num_classes, checkpointpath):
    model, preprocess, weights, classes = get_torchvision_detection_models(modelname)
    model = modify_fasterrcnnheader(model, num_classes, freeze=False)
    if checkpointpath:
        model.load_state_dict(torch.load(checkpointpath))
    return model, preprocess

# Function to modify the backbone for FasterRCNN
def modify_backbone(model, num_classes):
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

# Function to create test data
def create_testdata():
    images = torch.rand(4, 3, 600, 1200)
    boxes = torch.rand(4, 11, 4)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]  # Convert xywh to xyxy
    labels = torch.randint(1, 91, (4, 11))
    images = list(images)
    targets = [{'boxes': boxes[i], 'labels': labels[i]} for i in range(len(images))]
    return images, targets

# Function to test default models
def test_defaultmodels():
    model_names = list_models(module=torchvision.models)
    print("Torchvision built-in models:", model_names)
    detectionmodel_names = list_models(module=torchvision.models.detection)
    print("Torchvision detection models:", detectionmodel_names)

    modelname = 'fasterrcnn_resnet50_fpn_v2'
    model, preprocess, weights, classes = get_torchvision_detection_models(modelname, box_score_thresh=0.2)
    print(f"Backbone out_channels: {model.backbone.out_channels}")

    torch.save(model.state_dict(), "/data/cmpe249-fa23/modelzoo/fasterrcnn_resnet50_fpn_v2.pt")

    x = torch.rand(1, 3, 64, 64)
    output = model.backbone(x)
    print([(k, v.shape) for k, v in output.items()])

    module_list = list(model.named_children())
    for m in module_list:
        print(m[0])
        print(len(m))

    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    print(predictions)

    images, targets = create_testdata()
    output = model(images, targets)

    torch.onnx.export(model, x, "/data/cmpe249-fa23/trainoutput/faster_rcnn.onnx", opset_version=11)

    summary(
        model=model,
        input_size=(1, 3, 300, 400),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

# Function to find intersecting dictionary keys
def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

# Function to load checkpoint
def load_checkpoint(model, ckpt_file, fp16=False):
    ckpt = torch.load(ckpt_file, map_location='cpu')
    current_model_statedict = model.state_dict()
    csd = intersect_dicts(ckpt, current_model_statedict)
    model.load_state_dict(ckpt, strict=False)
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights')
    model.half() if fp16 else model.float()
    return model

# Function to create a detection model
def create_detectionmodel(modelname, num_classes=None, trainable_layers=0, ckpt_file=None, fp16=False, device='cuda:0', scale='n'):
    model = None
    preprocess = None
    classes = None
    freezemodel = trainable_layers == 0

    if modelname == 'fasterrcnn_resnet50_fpn_v2':
        model, preprocess, weights, classes = get_torchvision_detection_models(modelname)
        if num_classes is not None and len(classes) != num_classes:
            model = modify_fasterrcnnheader(model, num_classes, freeze=freezemodel)
        if ckpt_file:
            model = load_checkpoint(model, ckpt_file, fp16)
    elif modelname.startswith('customrcnn'):
        x = modelname.split("_")
        if x[0] == 'customrcnn' and x[1].startswith('resnet'):
            backbonename = x[1]
            model = CustomRCNN(
                backbone_modulename=backbonename,
                trainable_layers=trainable_layers,
                num_classes=num_classes,
                out_channels=256,
                min_size=800,
                max_size=1333
            )
            if ckpt_file:
                model = load_checkpoint(model, ckpt_file, fp16)
        else:
            print("Model name not supported")
    elif modelname.startswith('yolo'):
        model, preprocess, classes = create_yolomodel(modelname, num_classes, ckpt_file, fp16, device, scale)
        model = freeze_yolomodel(model, freeze=[])
    else:
        print('Model name not supported')

    if model:
        summary(
            model=model,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )
        if device:
            model = model.to(device)
    return model, preprocess, classes

if __name__ == "__main__":
    test_defaultmodels()

    os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    DATAPATH = '/data/cmpe249-fa23/torchvisiondata/'
