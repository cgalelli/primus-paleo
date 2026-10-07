"""
OptimusPrimus: two-stage track detection in optical-microscopy images.

A segmentation network (MAnet + EfficientNet-B7) proposes candidate tracks; a classifier (EfficientNet-B0 on 64x64
patches aligned to the track axis) removes false positives. Tracks are measured by ellipse fitting. Every z-stack is
analysed on its sharpest focal plane, as the annotated tiles are.

Workflow
    0. Data preparation       slice_tif_to_png_tiles, convert_xml_to_masks
    1. Segmentation           train_segmentation          -> Data/models/<SEG_NAME>/<SEG_NAME>.pth (+ .json, descriptor)
    2. Classifier patches     automatic: train_classifier(seg, patches='auto')
                              manual:    export_candidate_patches -> sort unsorted/ into track/ and bkg/
                                         -> train_classifier(seg, patches=<folder>)
    3. Classifier             train_classifier            -> Data/models/<SEG_NAME>/<SEG_NAME>__<CLS_TAG>.pth (+ .json, descriptor)
       (1-3 in one call: train_all, segmentation + automatic patches + classifier)
    4. Inference              run_inference(images, seg_model, cls_model=None)   (cls_model None: segmentation only)
    5. Detection model        load_detection_logs -> detection_descriptor -> detection_model -> fold_detection
                              -> poisson_chi2

Model folder Data/models/<SEG_NAME>/
    <SEG_NAME>.pth / .json / _training_metrics.csv           segmentation weights, training record (files, split,
                                                            hyperparameters), per-epoch metrics
    <SEG_NAME>_st<t>_detection_logs.json / _descriptor.csv  segmentation-only descriptor at threshold t
    <SEG_NAME>__<CLS_TAG>.pth / .json / _training_metrics.csv
    <SEG_NAME>__<CLS_TAG>_st<t>_ct<c>_detection_logs.json / _descriptor.csv
    patches_auto_pt<t>/                                     automatic classifier patches (cache, rewritten)
    patches_manual_<images>/                                default target of export_candidate_patches
"""
import os
import re
import glob
import json
import shutil
import hashlib
import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from skimage.measure import label, regionprops
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from scipy.stats import norm as _norm, chi2 as _chi2_dist
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_CONFIG = {
    'img_height': 1024,
    'img_width': 1024,
    'pixel_resolution_um_per_px': 0.345,
}

SEG_MODEL_CONFIG = {
    'model_arc': 'MAnet',
    'encoder': 'efficientnet-b7',
    'encoder_weights': 'imagenet',
    'threshold': 0.1,                       # operating point: patches, inference, descriptor (stored with the model)
}

CLASS_MODEL_CONFIG = {
    'encoder': 'efficientnet_b0',
    'pretrained': True,                     # ImageNet initialisation
    'threshold': 0.55,                      # operating point: inference, descriptor (stored with the model)
}

TRAINING_PARAMETERS = {
    'seg_epochs': 200,
    'seg_min_epochs': 100,
    'seg_batch_size': 2,
    'seg_learning_rate': 3e-4,
    'seg_patience': 30,
    'seg_threshold': 0.5,                   # threshold of the validation metrics during training
    'seg_metric_to_monitor': 'Dice/F1',     # one of 'FBeta', 'IoU', 'Dice/F1', 'Recall', 'Precision'
    'seg_val_beta': 2.0,
    'class_boost_precision_weight': 0.7,
    'class_learning_rate': 3e-5,
    'class_epochs': 80,
    'class_batch_size': 64,
    'class_patience': 2,
    'class_threshold': 0.5,                 # threshold of the validation metrics during training
}

TRAIN_IMAGE = {
    'mask_subdir': 'training_masks',
    'image_extensions': '*.png',
    'mask_extension': '_mask.png',
    'test_split_ratio': 0.1,
    'val_split_ratio': 0.15,
    'split_seed': 42,
}

AUGMENTATION_PARAMETERS = {
    'H_FLIP_PROB': 0.5,
    'V_FLIP_PROB': 0.5,
    'BRIGHTNESS_CONTRAST_PROB': 0.4,
}

MODELS_ROOT = "Data/models/"
INFERENCE_ROOT = "Data/inference_results/"
TILES_SUBDIR = "tiles"

MANUAL_PATCH_VAL_RATIO = 0.15               # validation share of hand-sorted patches without train/ and val/ subfolders
MIN_TRACK_AREA_PX = 10                      # smaller components are never reported (ellipse fit), nor used as patches
IOU_THRESHOLD = 0.5                         # matching of predictions to annotations (patch labels and descriptor)

DEFAULT_BIN_EDGES_NM = np.linspace(0., 16000., 17)   # grid of the descriptor table written at training time
DESCRIPTOR_MIN_COUNT = 5                    # bins with fewer annotated tracks borrow the recall of the nearest bin
DESCRIPTOR_MIN_PAIRS = 5                    # bins with fewer matched pairs borrow the length response of the nearest bin

# keep_patches_in_memory=True holds every 64x64x3 uint8 patch (~12 KB) in RAM: warn above this number
MAX_IN_MEMORY_PATCHES_WARNING = 50000

# z-stack tiles: Frame<n>_<acquisition>_<i>_<j>.png, grouped by (n, i, j)
_FRAME_RE = re.compile(r'^Frame(\d+)_.+_(\d+)_(\d+)\.png$')

_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ==========================================
# DATA PREPARATION
# ==========================================

def slice_tif_to_png_tiles(images_per_group, input_dir, output_spec=TILES_SUBDIR, tile_size=IMAGE_CONFIG['img_height']):
    """Slices TIFF acquisitions into square PNG tiles named Frame<group>_<file stem>_<i>_<j>.png.

    Consecutive files (sorted by name) form one z-stack group of `images_per_group` focal planes; only full-size tiles
    are written.

    Args:
        images_per_group (int): Number of focal planes per z-stack.
        input_dir (str): Folder with the TIFF files; the tiles go to input_dir/output_spec.
        output_spec (str, optional): Tiles subfolder. Defaults to TILES_SUBDIR.
        tile_size (int, optional): Tile side in pixels. Defaults to IMAGE_CONFIG['img_height'].

    Raises:
        ValueError: If the number of TIFF files is not a multiple of images_per_group.
    """
    output_dir = os.path.join(input_dir, output_spec)
    os.makedirs(output_dir, exist_ok=True)
    tif_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff')))
    if len(tif_files) % images_per_group != 0:
        raise ValueError(f"The number of images ({len(tif_files)}) is not a multiple of images_per_group ({images_per_group}).")
    print(f"Found {len(tif_files)} TIFF images, {len(tif_files) // images_per_group} groups of {images_per_group}.")

    for index, filename in enumerate(tif_files):
        group = index // images_per_group + 1
        base_name = os.path.splitext(filename)[0]
        try:
            img = Image.open(os.path.join(input_dir, filename))
            n_tiles = 0
            for i in range(img.width // tile_size):
                for j in range(img.height // tile_size):
                    box = (i * tile_size, j * tile_size, (i + 1) * tile_size, (j + 1) * tile_size)
                    img.crop(box).save(os.path.join(output_dir, f"Frame{group}_{base_name}_{i}_{j}.png"), "PNG")
                    n_tiles += 1
            print(f"{filename} (group {group}): {n_tiles} tiles")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


def convert_xml_to_masks(xml_path, output_dir):
    """Converts CVAT RLE mask annotations (label 'Track') into binary PNG masks named <image stem>_mask.png.

    Args:
        xml_path (str): CVAT XML annotation file.
        output_dir (str): Output folder for the masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_tags = ET.parse(xml_path).getroot().findall('image')
    if not image_tags:
        raise ValueError(f"No <image> tags found in '{xml_path}'.")

    for image_tag in image_tags:
        final_mask = np.zeros((int(image_tag.get('height')), int(image_tag.get('width'))), dtype=np.uint8)
        for mask_tag in image_tag.findall('mask'):
            if mask_tag.get('label') != 'Track':
                continue
            runs = [int(p) for p in mask_tag.get('rle').split(', ')]
            values = np.repeat(np.arange(len(runs)) % 2, runs).astype(np.uint8)      # runs alternate 0, 1, 0, ...
            height, width = int(mask_tag.get('height')), int(mask_tag.get('width'))
            top, left = int(mask_tag.get('top')), int(mask_tag.get('left'))
            roi = final_mask[top:top + height, left:left + width]
            np.maximum(roi, values.reshape(height, width) * 255, out=roi)
        stem = os.path.splitext(image_tag.get('name'))[0]
        cv2.imwrite(os.path.join(output_dir, f"{stem}_mask.png"), final_mask)
    print(f"Processed {len(image_tags)} images; masks saved to {output_dir}")


# ==========================================
# NAMING AND MODEL RECORDS
# ==========================================

def _slug(text):
    """Text reduced to letters, digits and . + - (other runs become '-')."""
    return re.sub(r'[^A-Za-z0-9.+-]+', '-', str(text)).strip('-')


def _short_hash(items):
    """6-character hash of a set of strings (order independent)."""
    return hashlib.md5('\n'.join(sorted(items)).encode()).hexdigest()[:6]


def seg_model_name(image_spec, n_tiles, file_hash, seg_cfg, params):
    """Folder and file name of a segmentation model: annotated set, architecture and training hyperparameters.

    Example: seg_ARCI-URAS26F2_n150-3fa2c1_MAnet-efficientnet-b7_ep200-bs2-lr3e-04-Dice-F1
    """
    return (f"seg_{_slug(image_spec)}_n{n_tiles}-{file_hash}_{seg_cfg['model_arc']}-{_slug(seg_cfg['encoder'])}"
            f"_ep{params['seg_epochs']}-bs{params['seg_batch_size']}-lr{params['seg_learning_rate']:.0e}"
            f"-{_slug(params['seg_metric_to_monitor'])}")


def cls_model_tag(source_tag, patch_th, cls_cfg, params):
    """Classifier part of a classifier file name: patch source and threshold, network and hyperparameters.

    Example: cls-auto_pt0.1_efficientnet-b0_ep80-bs64-lr3e-05-pw0.7
    """
    init = '' if cls_cfg['pretrained'] else '-scratch'
    return (f"cls-{source_tag}_pt{patch_th:g}_{_slug(cls_cfg['encoder'])}{init}"
            f"_ep{params['class_epochs']}-bs{params['class_batch_size']}-lr{params['class_learning_rate']:.0e}"
            f"-pw{params['class_boost_precision_weight']:g}")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serialisable")


def _save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def _record_path(pth):
    """Training record (json) of a checkpoint."""
    return os.path.splitext(pth)[0] + '.json'


def _load_record(pth, kind):
    record_path = _record_path(pth)
    if not os.path.exists(record_path):
        raise FileNotFoundError(f"Model record not found at '{record_path}' (models trained with this module have one).")
    with open(record_path) as f:
        record = json.load(f)
    if record.get('kind') != kind:
        raise ValueError(f"'{pth}' is a {record.get('kind')} model, not a {kind} model.")
    return record


def resolve_seg_model(seg_model):
    """Segmentation checkpoint and its training record.

    Args:
        seg_model (str): The model folder Data/models/<SEG_NAME> or the .pth file.

    Returns:
        tuple: (path of the .pth, record dict).
    """
    path = seg_model
    if os.path.isdir(path):
        path = os.path.join(path, os.path.basename(os.path.normpath(path)) + '.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segmentation model not found at '{path}'")
    return path, _load_record(path, 'segmentation')


def resolve_cls_model(cls_model, seg_record=None):
    """Classifier checkpoint and its training record; warns if it was trained on another segmentation model.

    Returns:
        tuple: (path of the .pth, record dict).
    """
    if not os.path.exists(cls_model):
        raise FileNotFoundError(f"Classification model not found at '{cls_model}'")
    record = _load_record(cls_model, 'classification')
    if seg_record is not None and record['seg_model'] != seg_record['name']:
        print(f"Warning: the classifier was trained on candidates of '{record['seg_model']}', "
              f"not of '{seg_record['name']}'.")
    return cls_model, record


def descriptor_paths(seg_pth, seg_th, cls_pth=None, cls_th=None):
    """Detection logs (json) and descriptor table (csv) of a model at given thresholds, in the model folder.

    Returns:
        tuple: (logs path, table path, base name).
    """
    model_dir = os.path.dirname(seg_pth)
    if cls_pth is None:
        base = f"{os.path.splitext(os.path.basename(seg_pth))[0]}_st{seg_th:g}"
    else:
        base = f"{os.path.splitext(os.path.basename(cls_pth))[0]}_st{seg_th:g}_ct{cls_th:g}"
    return (os.path.join(model_dir, base + '_detection_logs.json'), os.path.join(model_dir, base + '_descriptor.csv'), base)


def list_models(models_root=MODELS_ROOT):
    """Segmentation models under `models_root` and the classifiers trained on each.

    Returns:
        pandas.DataFrame: One row per model with kind, name, training summary and path.
    """
    rows = []
    for seg_dir in sorted(glob.glob(os.path.join(models_root, 'seg_*'))):
        for record_path in sorted(glob.glob(os.path.join(seg_dir, '*.json'))):
            if record_path.endswith(('_detection_logs.json', 'export_info.json')):
                continue
            with open(record_path) as f:
                rec = json.load(f)
            if rec.get('kind') not in ('segmentation', 'classification'):
                continue
            res = rec.get('training_result', {})
            rows.append({'seg_model': os.path.basename(seg_dir), 'kind': rec['kind'],
                         'classifier': '' if rec['kind'] == 'segmentation' else rec['tag'],
                         'threshold': rec['threshold'], 'best_epoch': res.get('best_epoch'),
                         'created': rec.get('created'), 'path': os.path.splitext(record_path)[0] + '.pth'})
    return pd.DataFrame(rows)


# ==========================================
# BUILDING BLOCKS
# ==========================================

def resolve_num_workers(parallel=True):
    """Number of CPU workers: a small pool for data loading when a GPU is available, otherwise all cores
    (or none if `parallel` is False)."""
    cpu_count = os.cpu_count() or 1
    if torch.cuda.is_available():
        return min(4, cpu_count // 2) if cpu_count > 1 else 0
    return cpu_count if parallel else 0


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_sharpness_score(fpath):
    """Focus score of an image: variance of the Laplacian.

    Returns:
        tuple: (fpath, score, error message or None); the score is 0 for unreadable or uniform images.
    """
    try:
        img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return fpath, 0.0, f"[ERROR] Unable to read file: {os.path.basename(fpath)}"
        score = 0.0 if np.var(img_gray) == 0 else cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return fpath, score, None
    except Exception as e:
        return fpath, 0.0, f"{os.path.basename(fpath)}: unexpected error: {e}"


def _images_tag(images):
    """Short name of an image source (the parent folder when the folder is the tiles subfolder)."""
    if isinstance(images, (list, tuple)):
        return f"list{len(images)}-{_short_hash([os.path.basename(p) for p in images])}"
    path = os.path.normpath(images)
    name = os.path.basename(path)
    if name == TILES_SUBDIR:
        name = os.path.basename(os.path.dirname(path))
    return _slug(os.path.splitext(name)[0] if os.path.isfile(path) else name)


def collect_images(images, parallel=True):
    """Images to analyse, one per z-stack: the sharpest focal plane of each group Frame<n>_*_<i>_<j>.png; files
    that do not follow the z-stack naming are taken one by one.

    Args:
        images (str or list): Folder of PNG tiles (or a folder containing the TILES_SUBDIR subfolder), one file,
            or a list of files.
        parallel (bool, optional): Spread the sharpness analysis over all CPU cores when there is no GPU.

    Returns:
        list: Image paths.
    """
    if isinstance(images, (list, tuple)):
        paths = sorted(images)
    elif os.path.isdir(images):
        paths = sorted(glob.glob(os.path.join(images, '*.png')))
        if not paths and os.path.isdir(os.path.join(images, TILES_SUBDIR)):
            paths = sorted(glob.glob(os.path.join(images, TILES_SUBDIR, '*.png')))
    elif os.path.isfile(images):
        paths = [images]
    else:
        raise FileNotFoundError(f"No images at '{images}'")
    if not paths:
        raise FileNotFoundError(f"No PNG images found in '{images}'")

    groups = {}
    for p in paths:
        m = _FRAME_RE.match(os.path.basename(p))
        groups.setdefault(f"Frame{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else p, []).append(p)
    stacked = [p for files in groups.values() if len(files) > 1 for p in files]
    if not stacked:
        return [files[0] for files in groups.values()]

    workers = resolve_num_workers(parallel)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(compute_sharpness_score, stacked, chunksize=8), total=len(stacked),
                                desc="Sharpest focal plane"))
    else:
        results = [compute_sharpness_score(p) for p in tqdm(stacked, desc="Sharpest focal plane")]
    scores = {}
    for path, score, error in results:
        if error:
            print(error)
        scores[path] = score
    print(f"{len(paths)} images in {len(groups)} z-stacks: the sharpest plane of each is used.")
    return [max(files, key=lambda p: scores.get(p, 0.)) if len(files) > 1 else files[0] for files in groups.values()]


def get_seg_preprocessing(encoder, encoder_weights, height=None, width=None):
    """Encoder normalisation (optionally preceded by a resize to height x width) and tensor conversion.

    Falls back to scaling to [0, 1] when the encoder has no normalisation settings for `encoder_weights`.
    """
    try:
        normalise = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    except (ValueError, KeyError):
        def normalise(image, **kwargs):
            return image.astype(np.float32) / 255.0
    resize = [A.Resize(height, width, interpolation=cv2.INTER_LINEAR)] if height and width else []
    return A.Compose(resize + [A.Lambda(image=normalise), ToTensorV2()])


def get_class_transform(train=False):
    """Transform of the 64x64 classifier patches (with flips and brightness/contrast jitter if `train`)."""
    augment = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3)] if train else []
    return A.Compose([A.Resize(64, 64)] + augment + [A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD), ToTensorV2()])


def build_seg_model(arch, encoder, encoder_weights=None, weights_path=None, device='cpu'):
    """Segmentation network (1 output channel, logits). With `weights_path` the checkpoint is loaded, otherwise the
    encoder is initialised with `encoder_weights`.

    Raises:
        FileNotFoundError: If `weights_path` is given but does not exist.
    """
    model = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=None if weights_path else encoder_weights,
                             in_channels=3, classes=1, activation=None)
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Segmentation checkpoint not found at '{weights_path}'")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device)


def build_class_model(model_type, weights_path=None, pretrained=True, device='cpu'):
    """Patch classifier: timm EfficientNet with stride 1 in the stem and in the first block of stage 2 (to keep
    resolution on 64x64 inputs) and a dropout + linear head with one logit.

    Raises:
        FileNotFoundError: If `weights_path` is given but does not exist.
    """
    model = timm.create_model(model_type, pretrained=pretrained and not weights_path, num_classes=1)
    model.conv_stem.stride = (1, 1)
    model.blocks[1][0].conv_dw.stride = (1, 1)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.classifier.in_features, 1))
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Classification checkpoint not found at '{weights_path}'")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state_dict.items()})
    return model.to(device)


def _load_seg(seg_pth, seg_record, device):
    """Trained segmentation network (eval mode) and its preprocessing, from the training record."""
    m, img = seg_record['model'], seg_record['image_config']
    model = build_seg_model(m['model_arc'], m['encoder'], weights_path=seg_pth, device=device).eval()
    return model, get_seg_preprocessing(m['encoder'], m['encoder_weights'], img['img_height'], img['img_width'])


def _load_cls(cls_pth, cls_record, device):
    """Trained classifier (eval mode), from the training record."""
    return build_class_model(cls_record['model']['encoder'], weights_path=cls_pth, device=device).eval()


def _parallelize(model):
    """Wraps the model in DataParallel when more than one GPU is available."""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        return nn.DataParallel(model)
    return model


def _state_dict(model):
    """State dict without the DataParallel 'module.' prefix."""
    return (model.module if isinstance(model, nn.DataParallel) else model).state_dict()


def predict_seg_mask(image_path, model, preprocessing, device, threshold):
    """Binary segmentation mask of one image.

    Returns:
        tuple: (RGB image, uint8 mask with 1 where sigmoid(logit) > threshold).
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    x = preprocessing(image=image)['image'].unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        mask = (torch.sigmoid(model(x)) > threshold).cpu().numpy().astype(np.uint8).squeeze()
    return image, mask


def candidate_regions(mask):
    """Connected components of a mask large enough to be reported as tracks (area >= MIN_TRACK_AREA_PX)."""
    return [r for r in regionprops(label(mask > 0)) if r.area >= MIN_TRACK_AREA_PX]


def get_64x64_centered_patch(img, mask, region):
    """64x64 RGB patch of one candidate, rotated so that its major axis is vertical, cropped to the object and
    centred on a black canvas (downscaled only if larger than 64 px).

    Args:
        img (np.ndarray): RGB image.
        mask (np.ndarray): Binary mask containing the candidate.
        region (RegionProperties): skimage region of the candidate.
    """
    cy, cx = region.centroid
    minr, minc, maxr, maxc = region.bbox
    diag = int(np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2)) + 10

    p_img = np.pad(img, ((diag, diag), (diag, diag), (0, 0)), mode='constant')
    p_mask = np.pad(mask, ((diag, diag), (diag, diag)), mode='constant')
    p_cy, p_cx = int(cy + diag), int(cx + diag)
    square_img = p_img[p_cy - diag:p_cy + diag, p_cx - diag:p_cx + diag]
    square_mask = p_mask[p_cy - diag:p_cy + diag, p_cx - diag:p_cx + diag]

    vertical_angle = np.rad2deg(-region.orientation) - 90
    rot_img = rotate(square_img, vertical_angle, resize=False, order=1, preserve_range=True).astype(np.uint8)
    rot_mask = rotate(square_mask, vertical_angle, resize=False, order=0, preserve_range=True)

    # keep only the component at the centre (or the one closest to it)
    rot_labeled = label(rot_mask > 0.5)
    h_rot, w_rot = rot_mask.shape
    center_label = rot_labeled[h_rot // 2, w_rot // 2]
    if center_label == 0:
        obj_coords = np.argwhere(rot_labeled > 0)
        if obj_coords.size > 0:
            closest = obj_coords[np.argmin(np.linalg.norm(obj_coords - np.array([h_rot // 2, w_rot // 2]), axis=1))]
            center_label = rot_labeled[closest[0], closest[1]]

    contours, _ = cv2.findContours((rot_labeled == center_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(contours[0])
    tight_obj_rgb = rot_img[y:y + h, x:x + w]
    h, w = tight_obj_rgb.shape[:2]
    if h > 64 or w > 64:
        scale = min(64 / h, 64 / w)
        tight_obj_rgb = cv2.resize(tight_obj_rgb, (max(1, int(w * scale)), max(1, int(h * scale))))   # >= 1 px for very thin objects
        h, w = tight_obj_rgb.shape[:2]

    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    start_y, start_x = (64 - h) // 2, (64 - w) // 2
    canvas[start_y:start_y + h, start_x:start_x + w] = tight_obj_rgb
    return canvas


def context_crop(img, region, half_size=64, scale=2):
    """Unrotated RGB crop around a candidate with its outline drawn, for sorting patches by eye.

    Returns:
        np.ndarray: RGB crop of side 2 * half_size * scale (larger for long candidates).
    """
    minr, minc, maxr, maxc = region.bbox
    half = max(half_size, int(0.75 * max(maxr - minr, maxc - minc)) + 8)
    cy, cx = (int(round(c)) for c in region.centroid)
    padded = np.pad(img, ((half, half), (half, half), (0, 0)), mode='constant')
    crop = padded[cy:cy + 2 * half, cx:cx + 2 * half].copy()
    comp = np.zeros(crop.shape[:2], np.uint8)
    rr, cc = region.coords[:, 0] - cy + half, region.coords[:, 1] - cx + half
    comp[rr, cc] = 1
    comp = cv2.dilate(comp, np.ones((7, 7), np.uint8))          # outline 3 px outside the candidate, which stays visible
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crop = cv2.resize(crop, (2 * half * scale, 2 * half * scale), interpolation=cv2.INTER_NEAREST)
    cv2.drawContours(crop, [c * scale for c in contours], -1, (0, 255, 0), 1)
    return crop


def classify_regions(image, mask, regions, model, transform, device, batch_size=256):
    """Track probability (sigmoid of the classifier logit) of each region of `mask`, in batches.

    Returns:
        list: One float per region.
    """
    if not regions:
        return []
    model.eval()
    probs = []
    with torch.no_grad():
        for k in range(0, len(regions), batch_size):
            batch = torch.stack([transform(image=get_64x64_centered_patch(image, mask, r))['image'] for r in regions[k:k + batch_size]])
            probs.append(torch.sigmoid(model(batch.to(device)))[:, 0].cpu())
    return torch.cat(probs).tolist()


def create_class_mask(image, mask, model, transform, device, threshold=0.5, batch_size=256):
    """Keeps only the connected components of `mask` that the classifier accepts (sigmoid > threshold).

    Returns:
        np.ndarray: Mask (same dtype as `mask`) with 1 on the accepted components.
    """
    labeled = label(mask)
    regions = regionprops(labeled)
    confirmed = np.zeros_like(mask)
    probs = classify_regions(image, mask, regions, model, transform, device, batch_size)
    confirmed[np.isin(labeled, [r.label for r, p in zip(regions, probs) if p > threshold])] = 1
    return confirmed


def label_candidates(gt_mask, pred_mask, iou_threshold=IOU_THRESHOLD):
    """Labels the segmentation candidates of one annotated image for the classifier.

    Every candidate (component of the predicted mask, see `candidate_regions`) is matched to the annotated tracks
    with the maximum-total-IoU assignment (Hungarian algorithm): target 1 if matched with IoU >= `iou_threshold`,
    0 otherwise. The patches are therefore those the classifier sees at inference, and the labels follow the
    matching of the descriptor. Annotated tracks missed by the segmentation give no sample.

    Returns:
        list: (region, target) pairs.
    """
    gt, pred = gt_mask > 0, pred_mask > 0
    gt_labels, pred_labels = label(gt), label(pred)
    gt_regions = regionprops(gt_labels)
    pred_regions = [r for r in regionprops(pred_labels) if r.area >= MIN_TRACK_AREA_PX]
    matched = np.zeros(len(pred_regions), bool)
    if gt_regions and pred_regions:
        width = pred_labels.max() + 1
        both = gt & pred
        inter = np.bincount(gt_labels[both] * width + pred_labels[both], minlength=(len(gt_regions) + 1) * width)
        inter = inter.reshape(len(gt_regions) + 1, width)[1:]
        inter = inter[:, [r.label for r in pred_regions]]
        area_gt = np.array([r.area for r in gt_regions])[:, None]
        area_pred = np.array([r.area for r in pred_regions])[None, :]
        iou = inter / (area_gt + area_pred - inter)
        for i, j in zip(*linear_sum_assignment(-iou)):
            matched[j] = iou[i, j] >= iou_threshold
    return [(r, int(m)) for r, m in zip(pred_regions, matched)]


def fit_ellipses(mask):
    """Ellipse fit of each outer contour of `mask` with area >= MIN_TRACK_AREA_PX and at least 5 points.

    Returns:
        list: Dicts with 'contour', 'center' (x, y), 'major_px', 'minor_px', 'angle_deg'.
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_TRACK_AREA_PX or len(contour) < 5:
            continue
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        ellipses.append({'contour': contour, 'center': (cx, cy), 'major_px': max(axis_a, axis_b),
                         'minor_px': min(axis_a, axis_b), 'angle_deg': angle})
    return ellipses


def _ellipse_records(mask, image_filename, um_per_px):
    """One record per fitted track of `mask`."""
    records = []
    for track_id, e in enumerate(fit_ellipses(mask), start=1):
        record = {"image_filename": image_filename, "track_id": track_id,
                  "centroid_x_px_ellipse": round(e['center'][0], 1), "centroid_y_px_ellipse": round(e['center'][1], 1),
                  "major_axis_px": round(e['major_px'], 2), "minor_axis_px": round(e['minor_px'], 2),
                  "orientation_deg": round(e['angle_deg'], 2)}
        if um_per_px is not None:
            record['len_um'] = round(e['major_px'] * um_per_px, 2)
        records.append(record)
    return records


def extract_instances(binary_mask, um_per_px=None):
    """Instances of a binary mask: size (ellipse major axis, in um if `um_per_px` is given) and filled mask, stored
    as a crop at its bounding box.

    Returns:
        list: Dicts with 'size', 'bbox' (x, y, w, h), 'mask' (bool crop) and 'area' (px).
    """
    instances = []
    for e in fit_ellipses(binary_mask):
        x, y, w, h = cv2.boundingRect(e['contour'])
        crop = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(crop, [e['contour'] - np.array([x, y])], -1, 1, thickness=cv2.FILLED)
        size = e['major_px'] * um_per_px if um_per_px is not None else e['major_px']
        instances.append({'size': size, 'bbox': (x, y, w, h), 'mask': crop.astype(bool), 'area': int(crop.sum())})
    return instances


def _instance_iou(a, b):
    """IoU of two instances from `extract_instances` (zero when the bounding boxes do not overlap)."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a['bbox'], b['bbox']
    x0, y0, x1, y1 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    inter = np.logical_and(a['mask'][y0 - ay:y1 - ay, x0 - ax:x1 - ax], b['mask'][y0 - by:y1 - by, x0 - bx:x1 - bx]).sum()
    return inter / (a['area'] + b['area'] - inter) if inter else 0.0


def match_instances_by_iou(gt_mask, pred_mask, iou_threshold=IOU_THRESHOLD, um_per_px=None):
    """Matches annotated and predicted instances with the assignment that maximises the total IoU (Hungarian
    algorithm), so that the result does not depend on the order of the instances; assigned pairs below
    `iou_threshold` are not matched.

    Returns:
        tuple: (gt_log, pred_log, pairs). The logs are lists of {'len_um', 'is_true_positive'} (the length is in px
            when `um_per_px` is None); pairs are the (annotated, reported) lengths of the matched instances.
    """
    gt, pred = extract_instances(gt_mask, um_per_px), extract_instances(pred_mask, um_per_px)
    gt_hit, pred_hit, pairs = np.zeros(len(gt), bool), np.zeros(len(pred), bool), []
    if gt and pred:
        iou = np.array([[_instance_iou(g, p) for p in pred] for g in gt])
        for i, j in zip(*linear_sum_assignment(-iou)):
            if iou[i, j] >= iou_threshold:
                gt_hit[i] = pred_hit[j] = True
                pairs.append((gt[i]['size'], pred[j]['size']))
    gt_log = [{'len_um': g['size'], 'is_true_positive': bool(hit)} for g, hit in zip(gt, gt_hit)]
    pred_log = [{'len_um': p['size'], 'is_true_positive': bool(hit)} for p, hit in zip(pred, pred_hit)]
    return gt_log, pred_log, pairs


def show_images(panels, title=None):
    """Shows images side by side; `panels` is a list of (image, subtitle)."""
    plt.figure(figsize=(4 * len(panels), 4))
    for k, (image, subtitle) in enumerate(panels, start=1):
        plt.subplot(1, len(panels), k)
        plt.imshow(image, cmap=None if image.ndim == 3 else 'gray')
        plt.title(subtitle)
        plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.show()


class SegmentationDataset(Dataset):
    """(image, mask) pairs for the segmentation network."""

    def __init__(self, file_pairs, augmentations, preprocessing):
        self.file_pairs = file_pairs
        self.augmentations = augmentations
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        augmented = self.augmentations(image=image, mask=mask)
        image_tensor = self.preprocessing(image=augmented['image'])['image']
        return image_tensor, torch.from_numpy(augmented['mask']).unsqueeze(0).float()


class TrackDataset(Dataset):
    """(patch, label) samples for the classifier; a patch is an RGB array or the path of a saved PNG."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, target = self.samples[idx]
        if isinstance(patch, str):
            patch = cv2.cvtColor(cv2.imread(patch), cv2.COLOR_BGR2RGB)
        return self.transform(image=patch)['image'], torch.tensor([target], dtype=torch.float32)


def _segmentation_scores(tp, fp, fn, beta):
    """Pixel-level (micro) scores from accumulated counts."""
    def ratio(a, b):
        return a / b if b else 0.0
    b2 = beta ** 2
    return {"FBeta": ratio((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp), "IoU": ratio(tp, tp + fp + fn),
            "Dice/F1": ratio(2 * tp, 2 * tp + fp + fn), "Recall": ratio(tp, tp + fn), "Precision": ratio(tp, tp + fp)}


# ==========================================
# TRAINING
# ==========================================

def _annotated_pairs(train_image_dir, train_cfg):
    """(image, mask) pairs of the annotated tiles."""
    mask_dir = os.path.join(train_image_dir, train_cfg['mask_subdir'])
    for path, what in ((train_image_dir, "Image directory"), (mask_dir, "Mask directory")):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{what} not found at '{path}'")
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(train_image_dir, train_cfg['image_extensions']))):
        mask_path = os.path.join(mask_dir, os.path.splitext(os.path.basename(img_path))[0] + train_cfg['mask_extension'])
        if os.path.exists(mask_path):
            pairs.append([img_path, mask_path])
    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs found in '{train_image_dir}' / '{mask_dir}'")
    return pairs


def _make_split(pairs, train_cfg):
    """Train / val / test split of the annotated pairs (deterministic for a given file list and seed)."""
    seed = train_cfg['split_seed']
    test_size, val_size = int(len(pairs) * train_cfg['test_split_ratio']), int(len(pairs) * train_cfg['val_split_ratio'])
    train_val, test = train_test_split(pairs, test_size=test_size, random_state=seed) if test_size > 0 else (pairs, [])
    train, val = (train_test_split(train_val, test_size=val_size, random_state=seed)
                  if 0 < val_size < len(train_val) else (train_val, []))
    if not val:
        raise ValueError(f"{len(pairs)} annotated tiles give an empty validation split: annotate more tiles.")
    return {'train': train, 'val': val, 'test': test}


def _held_out_paths(seg_record):
    """Absolute paths of the validation and test tiles of a segmentation model (used by the descriptor). Paths, not
    names: tile names such as Frame3_Acquisition_1_0_0.png repeat across samples."""
    return {os.path.abspath(img) for split in ('val', 'test') for img, _ in seg_record['split'][split]}


def train_segmentation(train_image_dir, image_spec, training_parameters=None, image_config=None, seg_model_config=None,
                       train_image=None, models_root=MODELS_ROOT, parallel=True, overwrite=False, describe=True):
    """Trains the segmentation network and stores it, with its training record and descriptor, in its own folder.

    The folder and file name (`seg_model_name`) summarise the annotated set and the hyperparameters; the record
    (<name>.json) holds the full configuration and the train/val/test split, which classifiers trained on this
    model reuse. With `describe`, the segmentation-only descriptor at the operating threshold is computed on the
    val + test tiles.

    Args:
        train_image_dir (str): Folder with the annotated tiles; masks in its `mask_subdir`.
        image_spec (str): Name of the annotated set (first part of the model name).
        training_parameters, image_config, seg_model_config, train_image (dict, optional): Overrides of the module
            defaults.
        models_root (str, optional): Parent folder of the model folders. Defaults to MODELS_ROOT.
        parallel (bool, optional): Without a GPU, load the data with all CPU cores.
        overwrite (bool, optional): Retrain if a model with the same name exists. Defaults to False.
        describe (bool, optional): Compute the descriptor after training. Defaults to True.

    Returns:
        str: Path of the trained checkpoint.

    Raises:
        FileExistsError: If the model exists and `overwrite` is False.
    """
    train_cfg = {**TRAIN_IMAGE, **(train_image or {})}
    img_cfg = {**IMAGE_CONFIG, **(image_config or {})}
    seg_cfg = {**SEG_MODEL_CONFIG, **(seg_model_config or {})}
    params = {**TRAINING_PARAMETERS, **(training_parameters or {})}

    pairs = _annotated_pairs(train_image_dir, train_cfg)
    split = _make_split(pairs, train_cfg)
    name = seg_model_name(image_spec, len(pairs), _short_hash([os.path.basename(p[0]) for p in pairs]), seg_cfg, params)
    model_dir = os.path.join(models_root, name)
    seg_pth = os.path.join(model_dir, name + '.pth')
    if os.path.exists(seg_pth):
        if not overwrite:
            raise FileExistsError(f"'{seg_pth}' exists: pass overwrite=True to retrain it.")
        n_cls = len(glob.glob(os.path.join(model_dir, f"{name}__*.pth")))
        if n_cls:
            print(f"Warning: {n_cls} classifier(s) in '{model_dir}' were trained on the previous weights.")
    os.makedirs(model_dir, exist_ok=True)

    record = {'kind': 'segmentation', 'name': name, 'created': _now(), 'image_spec': image_spec,
              'train_image_dir': train_image_dir, 'n_tiles': len(pairs),
              'n_split': {k: len(v) for k, v in split.items()}, 'split': split, 'train_image': train_cfg,
              'image_config': img_cfg,
              'model': {k: seg_cfg[k] for k in ('model_arc', 'encoder', 'encoder_weights')},
              'threshold': seg_cfg['threshold'],
              'training_parameters': {k: v for k, v in params.items() if k.startswith('seg_')},
              'augmentations': AUGMENTATION_PARAMETERS}
    print(f"[SEGMENTATION] {name}\n  {len(pairs)} annotated tiles: {record['n_split']}")
    record['training_result'] = _fit_segmentation(split['train'], split['val'], record, params, seg_pth, parallel)
    _save_json(_record_path(seg_pth), record)
    if describe:
        describe_model(seg_pth)
    print(f"[OUTPUT] segmentation model: {seg_pth}")
    return seg_pth


def _fit_segmentation(train_files, val_files, record, params, seg_pth, parallel):
    """Training loop of the segmentation network (0.5 Dice + 0.5 BCE on logits, AdamW, ReduceLROnPlateau on the
    monitored validation metric, early stopping after `seg_min_epochs`). Saves the best checkpoint at `seg_pth` and
    the per-epoch metrics next to it.

    Returns:
        dict: best_epoch, best_metric, epochs_run, interrupted.
    """
    p, m, img = params, record['model'], record['image_config']
    device = _device()
    preprocessing = get_seg_preprocessing(m['encoder'], m['encoder_weights'])
    resize = A.Resize(img['img_height'], img['img_width'], interpolation=cv2.INTER_LINEAR)
    train_augs = A.Compose([resize, A.HorizontalFlip(p=AUGMENTATION_PARAMETERS['H_FLIP_PROB']),
                            A.VerticalFlip(p=AUGMENTATION_PARAMETERS['V_FLIP_PROB']),
                            A.RandomBrightnessContrast(p=AUGMENTATION_PARAMETERS['BRIGHTNESS_CONTRAST_PROB'])])
    workers, pin = resolve_num_workers(parallel), torch.cuda.is_available()
    train_loader = DataLoader(SegmentationDataset(train_files, train_augs, preprocessing), batch_size=p['seg_batch_size'],
                              shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(SegmentationDataset(val_files, A.Compose([resize]), preprocessing), batch_size=p['seg_batch_size'],
                            shuffle=False, num_workers=workers, pin_memory=pin)

    model = _parallelize(build_seg_model(m['model_arc'], m['encoder'], m['encoder_weights'], device=device))
    dice_loss, bce_loss = smp.losses.DiceLoss(mode="binary", from_logits=True), smp.losses.SoftBCEWithLogitsLoss()

    def loss_fn(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    optimizer = optim.AdamW(model.parameters(), lr=p['seg_learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    metrics_csv = os.path.splitext(seg_pth)[0] + "_training_metrics.csv"
    best, best_epoch, counter, logs, interrupted = -1.0, 0, 0, [], False
    try:
        for epoch in range(p['seg_epochs']):
            model.train()
            train_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Seg train E{epoch + 1}", leave=False):
                images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                loss = loss_fn(model(images), masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss, counts = 0.0, np.zeros(3)            # pixel tp, fp, fn over the whole validation set
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
                    outputs = model(images)
                    val_loss += loss_fn(outputs, masks).item()
                    pred, truth = torch.sigmoid(outputs) > p['seg_threshold'], masks > 0.5
                    counts += [(pred & truth).sum().item(), (pred & ~truth).sum().item(), (~pred & truth).sum().item()]
            scores = _segmentation_scores(*counts, beta=p['seg_val_beta'])
            logs.append({"epoch": epoch + 1, "train_loss": train_loss / max(len(train_loader), 1),
                         "val_loss": val_loss / max(len(val_loader), 1), **{f"val_{k.lower()}": v for k, v in scores.items()}})
            monitored = scores[p['seg_metric_to_monitor']]
            print(f"Epoch {epoch + 1}: train loss {logs[-1]['train_loss']:.4f}, val loss {logs[-1]['val_loss']:.4f}, "
                  f"{p['seg_metric_to_monitor']} {monitored:.4f}")

            if monitored > best:
                best, best_epoch, counter = monitored, epoch + 1, 0
                torch.save(_state_dict(model), seg_pth)
                print(f"Saved segmentation checkpoint: {seg_pth}")
            else:
                counter += 1
                if counter >= p['seg_patience'] and epoch > p['seg_min_epochs']:
                    print("Early stopping triggered")
                    break
            scheduler.step(monitored)
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user.")
    finally:
        pd.DataFrame(logs).to_csv(metrics_csv, index=False)
        print(f"[INFO] Segmentation training finished. Metrics saved at: {metrics_csv}")
    if best_epoch == 0:
        raise RuntimeError("No segmentation checkpoint was saved.")
    return {'best_epoch': best_epoch, 'best_metric': best, 'epochs_run': len(logs), 'interrupted': interrupted}


def export_candidate_patches(seg_model, images, out_dir=None, seg_th=None, max_patches=None, context=True, seed=42,
                             overwrite=False, parallel=True):
    """Writes the classifier patches of the segmentation candidates of `images`, to be sorted by hand.

    Each candidate gives <out_dir>/unsorted/<tile>__c<k>.png, exactly the 64x64 patch the classifier sees, and,
    with `context`, <out_dir>/context/<tile>__c<k>.png, an unrotated view of the surroundings with the candidate
    outlined. Move the files of unsorted/ into track/ or bkg/ (created empty), then train with
    `train_classifier(seg_model, patches=out_dir)`. candidates.csv lists source tile, position and length of
    every patch; export_info.json records model and threshold.

    The validation and test tiles of the segmentation model are skipped: the descriptor is measured on them, and a
    classifier trained on their candidates would bias it.

    Args:
        seg_model (str): Segmentation model folder or .pth.
        images (str or list): Images to take candidates from (see `collect_images`; one plane per z-stack).
        out_dir (str, optional): Defaults to <model folder>/patches_manual_<images>.
        seg_th (float, optional): Segmentation threshold. Defaults to the model's operating point.
        max_patches (int, optional): Stop after this many patches, taking images in random order.
        context (bool, optional): Also write the context views. Defaults to True.
        seed (int, optional): Seed of the image order when `max_patches` is given.
        overwrite (bool, optional): Delete patches already in out_dir (sorted ones included). Defaults to False.

    Returns:
        str: out_dir.

    Raises:
        FileExistsError: If out_dir already holds patches and `overwrite` is False.
    """
    seg_pth, seg_record = resolve_seg_model(seg_model)
    seg_th = seg_record['threshold'] if seg_th is None else seg_th
    out_dir = out_dir or os.path.join(os.path.dirname(seg_pth), f"patches_manual_{_images_tag(images)}")
    subdirs = {k: os.path.join(out_dir, k) for k in ('unsorted', 'track', 'bkg', 'context')}
    existing = [p for k in ('unsorted', 'track', 'bkg') for p in glob.glob(os.path.join(subdirs[k], '**', '*.png'), recursive=True)]
    if existing and not overwrite:
        raise FileExistsError(f"'{out_dir}' already holds {len(existing)} patches: pass overwrite=True to replace them "
                              f"(sorted ones included) or choose another out_dir.")
    for d in subdirs.values():
        shutil.rmtree(d, ignore_errors=True)
        if d != subdirs['context'] or context:
            os.makedirs(d, exist_ok=True)

    files = collect_images(images, parallel)
    held_out = _held_out_paths(seg_record)
    n_skipped = sum(os.path.abspath(f) in held_out for f in files)
    files = [f for f in files if os.path.abspath(f) not in held_out]
    if n_skipped:
        print(f"{n_skipped} validation/test tiles of the segmentation model skipped.")
    if not files:
        raise ValueError("No images left after removing the validation/test tiles of the segmentation model.")
    if max_patches is not None:
        files = [files[k] for k in np.random.default_rng(seed).permutation(len(files))]

    device = _device()
    model, preprocessing = _load_seg(seg_pth, seg_record, device)
    um = seg_record['image_config']['pixel_resolution_um_per_px']
    rows, n_images = [], 0
    for path in tqdm(files, desc="Exporting candidate patches"):
        if max_patches is not None and len(rows) >= max_patches:
            break
        image, mask = predict_seg_mask(path, model, preprocessing, device, seg_th)
        n_images += 1
        stem = os.path.splitext(os.path.basename(path))[0]
        for region in candidate_regions(mask):
            if max_patches is not None and len(rows) >= max_patches:
                break
            fname = f"{stem}__c{region.label}.png"
            patch = get_64x64_centered_patch(image, mask, region)
            cv2.imwrite(os.path.join(subdirs['unsorted'], fname), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            if context:
                cv2.imwrite(os.path.join(subdirs['context'], fname), cv2.cvtColor(context_crop(image, region), cv2.COLOR_RGB2BGR))
            cy, cx = region.centroid
            rows.append({'patch': fname, 'source_image': path, 'centroid_x_px': round(cx, 1), 'centroid_y_px': round(cy, 1),
                         'area_px': int(region.area),
                         'len_um': round(region.axis_major_length * um, 2) if um else None})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'candidates.csv'), index=False)
    _save_json(os.path.join(out_dir, 'export_info.json'),
               {'seg_model': seg_record['name'], 'seg_th': seg_th, 'images': images if isinstance(images, str) else list(images),
                'n_images': n_images, 'n_patches': len(rows), 'created': _now()})
    print(f"[OUTPUT] {len(rows)} patches from {n_images} images in {subdirs['unsorted']}\n"
          f"  Next: move each file of unsorted/ into track/ or bkg/ (leave doubtful ones in unsorted/, they are not used),"
          f"\n  then train_classifier('{seg_pth}', patches='{out_dir}').")
    return out_dir


def _png_samples(folder, target):
    return [(p, target) for p in sorted(glob.glob(os.path.join(folder, '*.png')))]


def _folder_patches(patch_dir, seg_record):
    """Hand-sorted patches: <patch_dir>/track and /bkg (split train/val here), or <patch_dir>/train|val/track|bkg.

    Returns:
        tuple: (train samples, val samples, source dict for the record).
    """
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"Patch folder not found at '{patch_dir}'")
    info_path = os.path.join(patch_dir, 'export_info.json')
    info = None
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    if info and info['seg_model'] != seg_record['name']:
        print(f"Warning: the patches were exported with '{info['seg_model']}', not with '{seg_record['name']}'.")
    csv_path = os.path.join(patch_dir, 'candidates.csv')
    source_of = dict(pd.read_csv(csv_path)[['patch', 'source_image']].itertuples(index=False)) if os.path.exists(csv_path) else {}
    held_out = _held_out_paths(seg_record)

    def check_leak(samples):
        leaked = [p for p, _ in samples if os.path.abspath(str(source_of.get(os.path.basename(p), ''))) in held_out]
        if leaked:
            raise ValueError(f"{len(leaked)} patches come from validation/test tiles of the segmentation model "
                             f"(e.g. {os.path.basename(leaked[0])}): remove them, the descriptor is measured on those tiles.")

    if all(os.path.isdir(os.path.join(patch_dir, s)) for s in ('train', 'val')):
        train = _png_samples(os.path.join(patch_dir, 'train', 'track'), 1) + _png_samples(os.path.join(patch_dir, 'train', 'bkg'), 0)
        val = _png_samples(os.path.join(patch_dir, 'val', 'track'), 1) + _png_samples(os.path.join(patch_dir, 'val', 'bkg'), 0)
        split_mode = 'given'
    else:
        samples = _png_samples(os.path.join(patch_dir, 'track'), 1) + _png_samples(os.path.join(patch_dir, 'bkg'), 0)
        n_pos = sum(t for _, t in samples)
        if min(n_pos, len(samples) - n_pos) < 2:
            raise ValueError(f"'{patch_dir}' needs at least 2 patches in each of track/ and bkg/ ({n_pos} track, "
                             f"{len(samples) - n_pos} bkg): sort the files of unsorted/ first.")
        check_leak(samples)
        n_val = max(2, int(round(MANUAL_PATCH_VAL_RATIO * len(samples))))
        train, val = train_test_split(samples, test_size=n_val, random_state=TRAIN_IMAGE['split_seed'],
                                      stratify=[t for _, t in samples])
        split_mode = f'random {MANUAL_PATCH_VAL_RATIO:g}'
    check_leak(train + val)

    rel = [os.path.relpath(p, patch_dir) for p, _ in train + val]
    n_track = sum(t for _, t in train + val)
    source = {'type': 'manual', 'folder': patch_dir, 'split': split_mode, 'n_track': n_track,
              'n_bkg': len(rel) - n_track, 'hash': _short_hash(rel), 'export_info': info,
              'tag': f"manual-{_slug(os.path.basename(os.path.normpath(patch_dir)))}-n{len(rel)}-{_short_hash(rel)}"}
    return train, val, source


def _auto_patches(seg_pth, seg_record, seg_th, keep_patches_in_memory):
    """Automatically labelled patches (see `label_candidates`) of the train and val tiles of the segmentation model.
    Without `keep_patches_in_memory` they are written under <model folder>/patches_auto_pt<seg_th>/ (rewritten).

    Returns:
        tuple: (train samples, val samples, source dict for the record).
    """
    device = _device()
    model, preprocessing = _load_seg(seg_pth, seg_record, device)
    cache = os.path.join(os.path.dirname(seg_pth), f"patches_auto_pt{seg_th:g}")
    if not keep_patches_in_memory:
        shutil.rmtree(cache, ignore_errors=True)
    out, counts = {}, {}
    for split in ('train', 'val'):
        samples = []
        for img_path, mask_path in tqdm(seg_record['split'][split], desc=f"Patches ({split})", leave=False):
            image, pred_mask = predict_seg_mask(img_path, model, preprocessing, device, seg_th)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            for region, target in label_candidates(gt_mask, pred_mask):
                patch = get_64x64_centered_patch(image, pred_mask, region)
                if keep_patches_in_memory:
                    samples.append((patch, target))
                    continue
                folder = os.path.join(cache, split, 'track' if target else 'bkg')
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, f"{stem}__c{region.label}.png")
                cv2.imwrite(path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                samples.append((path, target))
        n_pos = sum(t for _, t in samples)
        counts[split] = {'track': n_pos, 'bkg': len(samples) - n_pos}
        print(f"[{split}] {n_pos} track / {len(samples) - n_pos} bkg candidates")
        out[split] = samples
    total = len(out['train']) + len(out['val'])
    if keep_patches_in_memory and total > MAX_IN_MEMORY_PATCHES_WARNING:
        print(f"Warning: {total} patches (~{total * 64 * 64 * 3 / 1024 ** 2:.0f} MB) are held in RAM; "
              f"consider keep_patches_in_memory=False.")
    source = {'type': 'auto', 'labelling': f'segmentation candidates, Hungarian IoU >= {IOU_THRESHOLD:g} with the annotations',
              'counts': counts, 'folder': None if keep_patches_in_memory else cache, 'tag': 'auto'}
    return out['train'], out['val'], source


def train_classifier(seg_model, patches='auto', training_parameters=None, class_model_config=None, seg_th=None,
                     keep_patches_in_memory=False, overwrite=False, describe=True):
    """Trains the patch classifier on the candidates of a segmentation model; stores it, with its record and
    descriptor, in the segmentation model folder as <SEG_NAME>__<CLS_TAG>.pth.

    Args:
        seg_model (str): Segmentation model folder or .pth.
        patches (str, optional): 'auto' (default): candidates of the model's train/val tiles, labelled by matching
            them to the annotations (`label_candidates`). Otherwise a folder of hand-sorted patches, with
            track/ and bkg/ (or train/ and val/, each with track/ and bkg/), e.g. from `export_candidate_patches`.
            The two sources are never mixed.
        training_parameters, class_model_config (dict, optional): Overrides of the module defaults.
        seg_th (float, optional): Segmentation threshold of the automatic candidates. Defaults to the model's
            operating point; for a patch folder, the threshold of its export is used.
        keep_patches_in_memory (bool, optional): Automatic patches in RAM instead of PNG files. Defaults to False.
        overwrite (bool, optional): Retrain if a classifier with the same name exists. Defaults to False.
        describe (bool, optional): Compute the segmentation + classification descriptor after training.

    Returns:
        str: Path of the trained checkpoint.
    """
    seg_pth, seg_record = resolve_seg_model(seg_model)
    cls_cfg = {**CLASS_MODEL_CONFIG, **(class_model_config or {})}
    params = {**TRAINING_PARAMETERS, **(training_parameters or {})}

    if isinstance(patches, str) and patches == 'auto':
        patch_th = seg_record['threshold'] if seg_th is None else seg_th
        train_samples, val_samples, source = _auto_patches(seg_pth, seg_record, patch_th, keep_patches_in_memory)
    else:
        train_samples, val_samples, source = _folder_patches(patches, seg_record)
        info = source['export_info']
        patch_th = info['seg_th'] if info else (seg_record['threshold'] if seg_th is None else seg_th)
    for split, samples in (('train', train_samples), ('val', val_samples)):
        n_pos = sum(t for _, t in samples)
        if n_pos == 0 or n_pos == len(samples):
            raise ValueError(f"The {split} samples have no {'track' if n_pos == 0 else 'bkg'} patches "
                             f"({len(samples)} samples): the classifier cannot be trained.")

    tag = cls_model_tag(source['tag'], patch_th, cls_cfg, params)
    cls_pth = os.path.join(os.path.dirname(seg_pth), f"{seg_record['name']}__{tag}.pth")
    if os.path.exists(cls_pth) and not overwrite:
        raise FileExistsError(f"'{cls_pth}' exists: pass overwrite=True to retrain it.")
    record = {'kind': 'classification', 'name': os.path.splitext(os.path.basename(cls_pth))[0], 'tag': tag,
              'created': _now(), 'seg_model': seg_record['name'], 'patch_threshold': patch_th,
              'source': {k: v for k, v in source.items() if k != 'tag'},
              'n_samples': {'train': len(train_samples), 'val': len(val_samples)},
              'model': {k: cls_cfg[k] for k in ('encoder', 'pretrained')}, 'threshold': cls_cfg['threshold'],
              'training_parameters': {k: v for k, v in params.items() if k.startswith('class_')}}
    print(f"[CLASSIFICATION] {record['name']}")
    record['training_result'] = _fit_classifier(train_samples, val_samples, cls_cfg, params, cls_pth)
    _save_json(_record_path(cls_pth), record)
    if describe:
        describe_model(seg_pth, cls_pth)
    print(f"[OUTPUT] classification model: {cls_pth}")
    return cls_pth


def _fit_classifier(train_samples, val_samples, cls_cfg, params, cls_pth):
    """Training loop of the classifier (weighted BCE, AdamW, ReduceLROnPlateau on the validation loss; the
    checkpoint with the best validation Dice is kept at `cls_pth`, the per-epoch metrics next to it).

    Returns:
        dict: best_epoch, best_val_dice, epochs_run.
    """
    p = params
    device = _device()
    train_loader = DataLoader(TrackDataset(train_samples, get_class_transform(train=True)), batch_size=p['class_batch_size'], shuffle=True)
    val_loader = DataLoader(TrackDataset(val_samples, get_class_transform()), batch_size=p['class_batch_size'], shuffle=False)

    model = _parallelize(build_class_model(cls_cfg['encoder'], pretrained=cls_cfg['pretrained'], device=device))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p['class_boost_precision_weight']], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=p['class_learning_rate'], weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=p['class_patience'])

    best_dice, best_epoch, logs, eps = -1.0, 0, [], 1e-8      # -1: the first epoch is always saved, even if its Dice is 0
    for epoch in range(p['class_epochs']):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Class train E{epoch + 1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, tp, fp, fn, tn = 0.0, 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += loss_fn(outputs, labels).item()
                pred, truth = torch.sigmoid(outputs) > p['class_threshold'], labels > 0.5
                tp += (pred & truth).sum().item()
                fp += (pred & ~truth).sum().item()
                fn += (~pred & truth).sum().item()
                tn += (~pred & ~truth).sum().item()
        val_loss /= max(len(val_loader), 1)
        logs.append({"epoch": epoch + 1, "train_loss": train_loss / max(len(train_loader), 1), "val_loss": val_loss,
                     "val_accuracy": (tp + tn) / (tp + tn + fp + fn + eps), "val_precision": tp / (tp + fp + eps),
                     "val_recall": tp / (tp + fn + eps), "val_iou": tp / (tp + fp + fn + eps),
                     "val_dice": 2 * tp / (2 * tp + fp + fn + eps)})
        scheduler.step(val_loss)
        if logs[-1]['val_dice'] > best_dice:
            best_dice, best_epoch = logs[-1]['val_dice'], epoch + 1
            torch.save(_state_dict(model), cls_pth)
            print(f"Epoch {epoch + 1}: val Dice {best_dice:.4f}, saved classification checkpoint: {cls_pth}")

    metrics_csv = os.path.splitext(cls_pth)[0] + "_training_metrics.csv"
    pd.DataFrame(logs).to_csv(metrics_csv, index=False)
    print(f"[INFO] Classification training finished. Metrics saved at: {metrics_csv}")
    return {'best_epoch': best_epoch, 'best_val_dice': best_dice, 'epochs_run': len(logs)}


def train_all(train_image_dir, image_spec, training_parameters=None, image_config=None, seg_model_config=None,
              class_model_config=None, train_image=None, models_root=MODELS_ROOT, parallel=True, overwrite=False,
              keep_patches_in_memory=False):
    """Segmentation training, automatic classifier patches and classifier training, each with its descriptor.

    Returns:
        tuple: (segmentation checkpoint, classification checkpoint).
    """
    seg_pth = train_segmentation(train_image_dir, image_spec, training_parameters, image_config, seg_model_config,
                                 train_image, models_root, parallel, overwrite)
    cls_pth = train_classifier(seg_pth, 'auto', training_parameters, class_model_config,
                               keep_patches_in_memory=keep_patches_in_memory, overwrite=overwrite)
    return seg_pth, cls_pth


# ==========================================
# DESCRIPTOR (measured on the annotated validation + test tiles)
# ==========================================

def describe_model(seg_model, cls_model=None, seg_th=None, cls_th=None, edges_nm=DEFAULT_BIN_EDGES_NM, visualize=False):
    """Runs the chain on the validation + test tiles of the segmentation model, matches the result to the annotations
    and writes, in the model folder, the per-instance detection logs (json) and the descriptor table on `edges_nm`
    (csv). Called automatically at the end of training; call it again for other thresholds.

    Args:
        seg_model (str): Segmentation model folder or .pth.
        cls_model (str, optional): Classifier .pth; None describes the segmentation-only chain.
        seg_th, cls_th (float, optional): Thresholds. Default to the operating points stored with the models.
        edges_nm (np.ndarray, optional): Bin edges of the csv table [nm].
        visualize (bool, optional): Show image, annotation and prediction of each tile.

    Returns:
        pandas.DataFrame: The descriptor table (see `detection_descriptor`).
    """
    seg_pth, seg_record = resolve_seg_model(seg_model)
    seg_th = seg_record['threshold'] if seg_th is None else seg_th
    cls_pth, cls_record = resolve_cls_model(cls_model, seg_record) if cls_model is not None else (None, None)
    if cls_record is not None:
        cls_th = cls_record['threshold'] if cls_th is None else cls_th
    mode = 'seg' if cls_pth is None else 'seg_class'

    device = _device()
    seg_net, preprocessing = _load_seg(seg_pth, seg_record, device)
    cls_net = _load_cls(cls_pth, cls_record, device) if cls_pth else None
    transform = get_class_transform()
    um = seg_record['image_config']['pixel_resolution_um_per_px']

    files = seg_record['split']['val'] + seg_record['split']['test']
    gt_log, pred_log, pair_log, area_cm2 = [], [], [], 0.0
    for img_path, mask_path in tqdm(files, desc=f"Descriptor ({mode})"):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image, pred_mask = predict_seg_mask(img_path, seg_net, preprocessing, device, seg_th)
        if cls_net is not None:
            pred_mask = create_class_mask(image, pred_mask, cls_net, transform, device, cls_th)
        g, r, pairs = match_instances_by_iou(gt_mask, pred_mask, IOU_THRESHOLD, um)
        gt_log += g
        pred_log += r
        pair_log += pairs
        if um is not None:
            area_cm2 += gt_mask.shape[0] * gt_mask.shape[1] * (um * 1e-4) ** 2
        if visualize:
            show_images([(image, "Input"), (gt_mask, "Annotation"), (pred_mask, mode)], title=os.path.basename(img_path))

    logs_path, table_path, base = descriptor_paths(seg_pth, seg_th, cls_pth, cls_th)
    meta = {'mode': mode, 'seg_model': seg_record['name'], 'seg_th': seg_th,
            'cls_model': cls_record['name'] if cls_record else None, 'cls_th': cls_th if cls_record else None,
            'iou_threshold': IOU_THRESHOLD, 'n_tiles': len(files), 'created': _now()}
    save_detection_logs(logs_path, gt_log, pred_log, pair_log, area_cm2 if um is not None else None,
                        'um' if um is not None else 'px', meta)
    n_tp = sum(r['is_true_positive'] for r in gt_log)
    n_tpp = sum(r['is_true_positive'] for r in pred_log)
    print(f"[{mode}] {len(files)} val+test tiles: recall {n_tp / max(len(gt_log), 1):.3f}, "
          f"precision {n_tpp / max(len(pred_log), 1):.3f} | TP {n_tp}, FP {len(pred_log) - n_tpp}, FN {len(gt_log) - n_tp}")
    if um is None:
        print("No pixel calibration: the descriptor table needs lengths in um and the area; only the logs were written.")
        return None
    table = detection_descriptor(load_detection_logs(logs_path), edges_nm)
    table.to_csv(table_path, index=False)
    print(f"[OUTPUT] descriptor: {table_path}")
    return table


def save_detection_logs(path, gt_log, pred_log, pair_log, val_area_cm2, length_unit='um', meta=None):
    """Writes the per-instance validation logs read by `load_detection_logs`.

    Args:
        path (str): Output json file.
        gt_log, pred_log (list): Annotated / reported instances with 'len_um' and 'is_true_positive'.
        pair_log (list): (annotated, reported) lengths of the matched pairs.
        val_area_cm2 (float or None): Total area of the validation images [cm^2] (None without pixel calibration).
        length_unit (str, optional): 'um' or 'px'. Defaults to 'um'.
        meta (dict, optional): Models and thresholds the logs refer to.
    """
    payload = {
        'meta': meta or {},
        'length_unit': length_unit,
        'val_area_cm2': None if val_area_cm2 is None else float(val_area_cm2),
        'gt_log': [{'len_um': float(r['len_um']), 'is_true_positive': bool(r['is_true_positive'])} for r in gt_log],
        'pred_log': [{'len_um': float(r['len_um']), 'is_true_positive': bool(r['is_true_positive'])} for r in pred_log],
        'pair_log': [[float(a), float(b)] for a, b in pair_log],
    }
    _save_json(path, payload)


def load_detection_logs(path):
    """Reads the validation logs written by `describe_model`.

    Returns:
        dict: Keys 'meta', 'gt_log', 'pred_log', 'pair_log', 'val_area_cm2' and 'length_unit'.
    """
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Detection logs not found ({path}): run describe_model for these models and thresholds.")
    with open(path, 'r') as f:
        logs = json.load(f)
    logs['pair_log'] = [tuple(p) for p in logs['pair_log']]
    return logs


def _length_pair_inliers(gt_len, pred_len, n_mad=5., max_iter=5):
    """Pairs kept for the length response: residuals of a linear bias fit within `n_mad` robust standard deviations
    (1.4826 MAD), iterated. Rejects gross ellipse-fit failures, e.g. on tracks cut by the tile border.

    Returns:
        np.ndarray: Boolean mask of the kept pairs.
    """
    keep = np.ones(len(gt_len), bool)
    if len(gt_len) < 10:
        return keep
    resid = pred_len - gt_len
    for _ in range(max_iter):
        b1, b0 = np.polyfit(gt_len[keep], resid[keep], 1)
        r = resid - (b0 + b1 * gt_len)
        scale = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new_keep = np.abs(r) <= n_mad * scale if scale > 0 else keep
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep


def _nearest_populated(ok):
    """Index of each bin itself if `ok`, else of the nearest bin that is."""
    good = np.where(ok)[0]
    if len(good) == 0:
        raise ValueError("No bin is populated enough: use coarser bins.")
    return np.array([i if ok[i] else good[np.argmin(np.abs(good - i))] for i in range(len(ok))])


def detection_descriptor(logs, edges_nm=DEFAULT_BIN_EDGES_NM, min_count=DESCRIPTOR_MIN_COUNT, min_pairs=DESCRIPTOR_MIN_PAIRS):
    """Binned description of the detection chain on the annotated validation + test tiles.

    Per bin of annotated (true) length: G annotated tracks, TP detected, FN missed, recall = TP/G, fn_rate = FN/G,
    and the length response of the matched pairs, bias = mean(reported - annotated) and sigma = std. Per bin of
    reported length: Q reported tracks, FP false, precision, fp_per_cm2 = FP / validation area.

    The *_fold columns are what `detection_model` uses: recall and length response of bins with fewer than
    `min_count` annotated tracks / `min_pairs` matched pairs are taken from the nearest populated bin (marked in
    'borrowed_recall' / 'borrowed_response'). Gross length outliers are excluded from the response first.

    Args:
        logs (dict): Output of `load_detection_logs`.
        edges_nm (np.ndarray, optional): Bin edges [nm], shared with the theoretical spectrum and the measurement.

    Returns:
        pandas.DataFrame: One row per bin; attrs hold 'area_cm2', 'n_pairs', 'n_pairs_rejected' and 'meta'.
    """
    if logs['length_unit'] != 'um' or logs['val_area_cm2'] is None:
        raise ValueError("The descriptor needs logs with lengths in um and the validation area (pixel calibration).")
    edges = np.asarray(edges_nm, float)
    nb = len(edges) - 1
    g_len = np.array([r['len_um'] for r in logs['gt_log']], float) * 1e3
    g_tp = np.array([r['is_true_positive'] for r in logs['gt_log']], bool)
    p_len = np.array([r['len_um'] for r in logs['pred_log']], float) * 1e3
    p_tp = np.array([r['is_true_positive'] for r in logs['pred_log']], bool)
    pairs = np.asarray(logs['pair_log'], float).reshape(-1, 2) * 1e3
    if len(pairs) == 0:
        raise ValueError("No matched pairs in the logs.")

    def binned(x, weights=None):
        idx = np.digitize(x, edges) - 1
        ok = (idx >= 0) & (idx < nb)
        return np.bincount(idx[ok], weights=None if weights is None else weights[ok], minlength=nb)

    G, TP = binned(g_len), binned(g_len[g_tp])
    Q, TPp = binned(p_len), binned(p_len[p_tp])
    FN, FP = G - TP, Q - TPp
    area = float(logs['val_area_cm2'])

    inliers = _length_pair_inliers(pairs[:, 0], pairs[:, 1])
    true_len, resid = pairs[inliers, 0], pairs[inliers, 1] - pairs[inliers, 0]
    n_pairs = binned(true_len)
    s1, s2 = binned(true_len, resid), binned(true_len, resid ** 2)

    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.where(G > 0, TP / G, np.nan)
        precision = np.where(Q > 0, TPp / Q, np.nan)
        bias = np.where(n_pairs > 0, s1 / n_pairs, np.nan)
        sigma = np.where(n_pairs > 1, np.sqrt(np.maximum(s2 - n_pairs * bias ** 2, 0.) / (n_pairs - 1)), np.nan)

    src_r = _nearest_populated(G >= min_count)
    src_m = _nearest_populated(n_pairs >= max(min_pairs, 2))
    table = pd.DataFrame({
        'bin_lo_nm': edges[:-1], 'bin_hi_nm': edges[1:],
        'G': G, 'TP': TP, 'FN': FN, 'recall': recall, 'fn_rate': 1. - recall,
        'Q': Q, 'FP': FP, 'precision': precision, 'fp_per_cm2': FP / area,
        'n_pairs': n_pairs, 'bias_nm': bias, 'sigma_nm': sigma,
        'recall_fold': recall[src_r], 'bias_fold_nm': bias[src_m], 'sigma_fold_nm': sigma[src_m],
        'borrowed_recall': src_r != np.arange(nb), 'borrowed_response': src_m != np.arange(nb)})
    table.attrs = {'area_cm2': area, 'n_pairs': int(inliers.sum()), 'n_pairs_rejected': int((~inliers).sum()),
                   'meta': logs.get('meta', {})}
    return table


# ==========================================
# DETECTION MODEL: mu_j = sum_i M_ji R_i N_i + phi_j A
# ==========================================

def migration_matrix(edges_nm, bias_nm, sigma_nm, n_sub=5):
    """Length migration M[j, i] = P(reported in bin j | true length in bin i): a Gaussian of mean L + bias_i and
    standard deviation sigma_i, averaged over `n_sub` points L of the true bin. Columns sum to at most one (the
    rest migrates outside the histogram).

    Returns:
        np.ndarray: Matrix (n_bins, n_bins).
    """
    edges = np.asarray(edges_nm, float)
    nb = len(edges) - 1
    M = np.zeros((nb, nb))
    for i in range(nb):
        L = edges[i] + (np.arange(n_sub) + 0.5) / n_sub * (edges[i + 1] - edges[i])
        cdf = _norm.cdf((edges[:, None] - (L + bias_nm[i])[None, :]) / max(sigma_nm[i], 1.0))
        M[:, i] = np.diff(cdf, axis=0).mean(axis=1)
    return M


def detection_model(descriptor):
    """Detection model built from a descriptor table: recall R_i, migration M_ji and false-positive rate phi_j.

    Returns:
        dict: edges [nm], R, M, phi [per cm^2 per bin].
    """
    edges = np.append(descriptor['bin_lo_nm'].to_numpy(), descriptor['bin_hi_nm'].iloc[-1])
    return {'edges': edges, 'R': descriptor['recall_fold'].to_numpy(),
            'M': migration_matrix(edges, descriptor['bias_fold_nm'].to_numpy(), descriptor['sigma_fold_nm'].to_numpy()),
            'phi': descriptor['fp_per_cm2'].to_numpy()}


def fold_detection(counts_true, model, area_cm2, return_parts=False):
    """Expected reported histogram mu_j = sum_i M_ji R_i N_i + phi_j A.

    Args:
        counts_true (np.ndarray): Sliced spectrum N_i, expected tracks per bin of `model['edges']` on the analysed
            area (e.g. from `slice_spectrum`).
        model (dict): Output of `detection_model`.
        area_cm2 (float): Analysed area A of the measured sample [cm^2].
        return_parts (bool, optional): Also return the true-track and false-positive parts.

    Returns:
        np.ndarray or tuple: mu, or (mu, true-track part, false-positive part).
    """
    N = np.asarray(counts_true, float)
    if len(N) != len(model['R']):
        raise ValueError(f"The spectrum has {len(N)} bins, the detection model {len(model['R'])}: use the same edges.")
    signal = model['M'] @ (model['R'] * N)
    fp = model['phi'] * area_cm2
    return (signal + fp, signal, fp) if return_parts else signal + fp


def poisson_chi2(observed, mu):
    """Poisson likelihood-ratio chi2 (Baker-Cousins), 2 sum[mu - n + n ln(n / mu)], valid at low counts without
    merging bins. Bins with mu = 0 and n = 0 are skipped. ndof is the number of bins used: subtract the parameters
    fitted to the data (e.g. 1 for a free normalisation).

    Returns:
        tuple: (chi2, ndof, p-value).
    """
    n, mu = np.asarray(observed, float), np.asarray(mu, float)
    use = (mu > 0) | (n > 0)
    n, mu = n[use], np.maximum(mu[use], 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = mu - n + np.where(n > 0, n * np.log(n / mu), 0.)
    chi2 = float(2. * terms.sum())
    return chi2, int(use.sum()), float(_chi2_dist.sf(chi2, int(use.sum())))


# ==========================================
# INFERENCE
# ==========================================

@dataclass
class InferenceResult:
    """Tracks found by `run_inference` and what produced them."""
    tracks: pd.DataFrame
    mode: str                       # 'seg' or 'seg_class'
    area_cm2: float
    n_images: int
    images: object
    seg_model: str
    seg_th: float
    cls_model: object               # None for a segmentation-only run
    cls_th: object
    um_per_px: float
    detection_logs: object          # descriptor logs of these models and thresholds (None if not computed)
    csv_path: object

    @property
    def lengths_nm(self):
        return self.tracks['len_um'].to_numpy() * 1e3

    def histogram(self, edges_nm):
        """Measured counts n_j in the bins `edges_nm`."""
        return np.histogram(self.lengths_nm, bins=edges_nm)[0]

    def summary(self):
        """Number of tracks, analysed area, density and length quartiles."""
        L = self.tracks['len_um'] if len(self.tracks) else pd.Series(dtype=float)
        return {'mode': self.mode, 'n_images': self.n_images, 'n_tracks': len(self.tracks), 'area_cm2': self.area_cm2,
                'density_per_cm2': len(self.tracks) / self.area_cm2 if self.area_cm2 else None,
                'len_um_mean': L.mean(), 'len_um_q25': L.quantile(0.25), 'len_um_median': L.median(),
                'len_um_q75': L.quantile(0.75)}


_TRACK_COLUMNS = ["image_filename", "track_id", "centroid_x_px_ellipse", "centroid_y_px_ellipse", "major_axis_px",
                  "minor_axis_px", "orientation_deg", "len_um"]


def run_inference(images, seg_model, cls_model=None, seg_th=None, cls_th=None, um_per_px=None,
                  output_dir=INFERENCE_ROOT, parallel=True, visualize=False):
    """Detects tracks on `images`: segmentation, then (if `cls_model` is given) classification of the candidates.
    With `cls_model=None` the run is segmentation only.

    Every z-stack is analysed on its sharpest focal plane (`collect_images`). Architectures, encoders, image size,
    pixel resolution and default thresholds come from the records stored with the models. The tracks are saved as
    <output_dir>/<images>__<model>_st<t>[_ct<c>].csv with a .json of the run, and the descriptor logs of the same
    models and thresholds are attached when they exist.

    Args:
        images (str or list): Folder of tiles (or its parent with a TILES_SUBDIR subfolder), a file or a list of files.
        seg_model (str): Segmentation model folder or .pth (mandatory).
        cls_model (str, optional): Classifier .pth; None for a segmentation-only run.
        seg_th, cls_th (float, optional): Thresholds. Default to the operating points stored with the models.
        um_per_px (float, optional): Pixel size. Defaults to the one the segmentation model was trained with.
        output_dir (str, optional): Defaults to INFERENCE_ROOT; None does not save.
        parallel (bool, optional): Parallel sharpness analysis without a GPU.
        visualize (bool, optional): Show each image with its final mask.

    Returns:
        InferenceResult
    """
    seg_pth, seg_record = resolve_seg_model(seg_model)
    seg_th = seg_record['threshold'] if seg_th is None else seg_th
    cls_pth, cls_record = resolve_cls_model(cls_model, seg_record) if cls_model is not None else (None, None)
    if cls_record is not None:
        cls_th = cls_record['threshold'] if cls_th is None else cls_th
    else:
        cls_th = None
    mode = 'seg' if cls_pth is None else 'seg_class'
    um = seg_record['image_config']['pixel_resolution_um_per_px'] if um_per_px is None else um_per_px

    files = collect_images(images, parallel)
    print(f"[{mode}] {len(files)} images | segmentation: {seg_record['name']} (th {seg_th:g})"
          + (f" | classifier: {cls_record['tag']} (th {cls_th:g})" if cls_record else " | no classifier"))
    device = _device()
    seg_net, preprocessing = _load_seg(seg_pth, seg_record, device)
    cls_net = _load_cls(cls_pth, cls_record, device) if cls_pth else None
    transform = get_class_transform()

    records, area_cm2 = [], 0.0
    for path in tqdm(files, desc=f"Inference ({mode})"):
        image, mask = predict_seg_mask(path, seg_net, preprocessing, device, seg_th)
        if cls_net is not None:
            mask = create_class_mask(image, mask, cls_net, transform, device, cls_th)
        records += _ellipse_records(mask, os.path.basename(path), um)
        if um is not None:
            area_cm2 += image.shape[0] * image.shape[1] * (um * 1e-4) ** 2
        if visualize:
            show_images([(image, os.path.basename(path)), (mask, mode)])
    tracks = pd.DataFrame(records, columns=_TRACK_COLUMNS if um is not None else _TRACK_COLUMNS[:-1])

    logs_path, _, base = descriptor_paths(seg_pth, seg_th, cls_pth, cls_th)
    if not os.path.exists(logs_path):
        print(f"Note: no descriptor for these models and thresholds; run describe_model to create '{logs_path}'.")
        logs_path = None
    csv_path = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{_images_tag(images)}__{base}.csv")
        tracks.to_csv(csv_path, index=False)
    result = InferenceResult(tracks, mode, area_cm2 if um is not None else None, len(files),
                             images if isinstance(images, str) else list(images), seg_pth, seg_th, cls_pth, cls_th, um,
                             logs_path, csv_path)
    if csv_path:
        meta = {k: v for k, v in result.__dict__.items() if k != 'tracks'}
        _save_json(os.path.splitext(csv_path)[0] + '.json', meta)
    area = f"{area_cm2:.4f} cm^2" if um is not None else "unknown area"
    print(f"[OUTPUT] {len(tracks)} tracks on {area}" + (f" -> {csv_path}" if csv_path else ""))
    return result


def load_inference(csv_path):
    """Reloads a run saved by `run_inference` (the csv and its json).

    Returns:
        InferenceResult
    """
    meta_path = os.path.splitext(csv_path)[0] + '.json'
    for path in (csv_path, meta_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' not found")
    with open(meta_path) as f:
        meta = json.load(f)
    return InferenceResult(tracks=pd.read_csv(csv_path), **meta)
