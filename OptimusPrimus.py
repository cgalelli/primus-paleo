"""
OptimusPrimus: two-stage track detection in optical-microscopy images.

A segmentation network (MAnet + EfficientNet-B7) proposes candidate tracks with high recall on each focal plane of a
z-stack; a classifier (EfficientNet-B0 on 64x64 patches aligned to the track axis) removes false positives. Tracks are
measured by ellipse fitting.

Contents
    Configuration
    Data preparation          slice_tif_to_png_tiles, convert_xml_to_masks
    Building blocks           naming, preprocessing, model builders, prediction, instances and matching
    Detection model           recall, length response and false positives for folding theoretical spectra
    OptimusPrimus             inference on z-stacks of image tiles
    OptimusPrimusTraining     training and evaluation of the two networks
"""
import os
import re
import glob
import json
import shutil
import xml.etree.ElementTree as ET
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
from scipy.optimize import nnls as _nnls, linear_sum_assignment

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
    'image_folder_path': "Data/images/",
}

SEG_MODEL_CONFIG = {
    'model_arc': 'MAnet',
    'encoder': 'efficientnet-b7',
    'encoder_weights': 'imagenet',
    'threshold': 0.1,                       # operating point: inference, evaluation, classifier-patch extraction
    'model_folder_path': "Data/models/segmentation/",
}

CLASS_MODEL_CONFIG = {
    'encoder': 'efficientnet_b0',
    'pretrained': True,                     # ImageNet initialisation when training a new classifier
    'threshold': 0.55,                      # operating point: inference and evaluation
    'model_folder_path': "Data/models/classification/",
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
    'data_split_subdir': 'data_split',
    'image_extensions': '*.png',
    'mask_extension': '_mask.png',
    'test_split_ratio': 0.1,
    'val_split_ratio': 0.15,
    'split_filename': 'split_set.json',
}

AUGMENTATION_PARAMETERS = {
    'H_FLIP_PROB': 0.5,
    'V_FLIP_PROB': 0.5,
    'BRIGHTNESS_CONTRAST_PROB': 0.4,
}

# keep_patches_in_memory=True holds every 64x64x3 uint8 patch (~12 KB) in RAM: warn above this number
MAX_IN_MEMORY_PATCHES_WARNING = 50000

TILES_SUBDIR = "tiles"

INFERENCE_OUTPUT_FOLDERS = {
    'seg': "Data/inference_results/segmentation/",
    'seg_class': "Data/inference_results/seg_class/",
}

MODES = ('seg', 'seg_class')

_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")


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
# BUILDING BLOCKS
# ==========================================

def make_seg_model_spec(encoder, image_spec):
    """Name (without extension) of a segmentation checkpoint trained on `image_spec`."""
    return f"seg_model_{encoder}_{image_spec}"


def make_class_model_spec(model_type, image_spec):
    """Name (without extension) of a classification checkpoint trained on `image_spec`."""
    return f"class_model_{model_type}_{image_spec}"


def efficiency_csv_paths(seg_spec, cls_spec, seg_folder, cls_folder):
    """Binned-efficiency csv of each mode: 'seg' next to the segmentation model, 'seg_class' next to the classifier."""
    return {'seg': os.path.join(seg_folder, f"binned_efficiency_{seg_spec}.csv"),
            'seg_class': os.path.join(cls_folder, f"binned_efficiency_{seg_spec}_{cls_spec}.csv")}


def detection_logs_file(efficiency_csv):
    """Validation logs of the detection model, saved next to the binned-efficiency csv."""
    return os.path.splitext(efficiency_csv)[0] + "_detection_logs.json"


def resolve_num_workers(parallel=True):
    """Number of CPU workers: a small pool for data loading when a GPU is available, otherwise all cores
    (or none if `parallel` is False)."""
    cpu_count = os.cpu_count() or 1
    if torch.cuda.is_available():
        return min(4, cpu_count // 2) if cpu_count > 1 else 0
    return cpu_count if parallel else 0


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


def predict_stack_mask(file_list, model, preprocessing, device, threshold):
    """Union (pixel-wise OR) of the segmentation masks of the focal planes of one z-stack.

    Returns:
        tuple: (RGB image of the first file, aggregated uint8 mask).
    """
    first_image, masks = None, []
    for path in file_list:
        image, mask = predict_seg_mask(path, model, preprocessing, device, threshold)
        if first_image is None:
            first_image = image
        masks.append(mask)
    return first_image, np.max(np.stack(masks), axis=0).astype(np.uint8)


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


NEGATIVES_MODES = ('rims', 'no_rims', 'unmatched')


def class_training_candidates(gt_mask, pred_mask, negatives='rims', iou_threshold=0.5):
    """Samples of one image for the classifier, as (mask, region, target, touches_annotation) tuples: the patch is
    built from `mask` around `region`, with target 1 (track) or 0 (background).

    negatives:
        'rims'      positives: the annotated tracks (annotation crops). Negatives: the parts of the predicted mask
                    outside the annotations, i.e. rims around real tracks and false-positive components.
        'no_rims'   as 'rims', but only the predicted components that do not touch an annotation are negatives.
        'unmatched' every predicted component is a sample, labelled with the IoU matching of the evaluation: positive
                    if matched to an annotated track, negative otherwise. Annotated tracks missed by the
                    segmentation are not used. The patches are those the classifier sees at inference.
    `touches_annotation` tells whether a negative lies within 1 px of an annotation (diagnostic).

    Raises:
        ValueError: If `negatives` is not one of NEGATIVES_MODES.
    """
    if negatives not in NEGATIVES_MODES:
        raise ValueError(f"negatives must be one of {NEGATIVES_MODES}, got '{negatives}'")
    gt, pred = gt_mask > 0, pred_mask > 0
    near_gt = cv2.dilate(gt.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gt_labels, pred_labels = label(gt), label(pred)

    def touches(region):
        return bool(near_gt[region.coords[:, 0], region.coords[:, 1]].any())

    if negatives == 'unmatched':
        gt_regions, pred_regions = regionprops(gt_labels), regionprops(pred_labels)
        matched = np.zeros(len(pred_regions), bool)
        if gt_regions and pred_regions:
            width = len(pred_regions) + 1
            both = gt & pred
            inter = np.bincount(gt_labels[both] * width + pred_labels[both], minlength=(len(gt_regions) + 1) * width)
            inter = inter.reshape(len(gt_regions) + 1, width)[1:, 1:]
            area_gt = np.array([r.area for r in gt_regions])[:, None]
            area_pred = np.array([r.area for r in pred_regions])[None, :]
            iou = inter / (area_gt + area_pred - inter)
            for i, j in zip(*linear_sum_assignment(-iou)):
                matched[j] = iou[i, j] >= iou_threshold
        return [(pred_mask, r, int(m), touches(r)) for r, m in zip(pred_regions, matched)]

    candidates = [(gt_mask, r, 1, True) for r in regionprops(gt_labels)]
    for r in regionprops(label(pred & ~gt) if negatives == 'rims' else pred_labels):
        near = touches(r)
        if negatives == 'rims' or not near:
            candidates.append((pred_mask, r, 0, near))
    return candidates


def fit_ellipses(mask):
    """Ellipse fit of each outer contour of `mask` with area >= 10 px and at least 5 points.

    Returns:
        list: Dicts with 'contour', 'center' (x, y), 'major_px', 'minor_px', 'angle_deg'.
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        if cv2.contourArea(contour) < 10 or len(contour) < 5:
            continue
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        ellipses.append({'contour': contour, 'center': (cx, cy), 'major_px': max(axis_a, axis_b),
                         'minor_px': min(axis_a, axis_b), 'angle_deg': angle})
    return ellipses


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


def match_instances_by_iou(gt_mask, pred_mask, iou_threshold=0.5, um_per_px=None):
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


def binned_efficiency_table(gt_log, pred_log, num_bins=10, unit='um'):
    """Recall (per annotated length) and precision (per reported length) in equal-population bins of the annotated
    lengths.

    Returns:
        tuple: (table, df_gt, df_pred, bin edges).

    Raises:
        ValueError: If there are no annotated or no reported instances.
    """
    df_gt, df_pred = pd.DataFrame(gt_log), pd.DataFrame(pred_log)
    if df_gt.empty or df_pred.empty:
        raise ValueError("No annotated or no reported instances: the efficiency cannot be computed.")
    df_gt['size_bin'], bins = pd.qcut(df_gt['len_um'], q=num_bins, retbins=True, duplicates='drop')
    df_pred['size_bin'] = pd.cut(df_pred['len_um'], bins=bins, include_lowest=True)
    table = pd.concat([
        df_gt.groupby('size_bin', observed=False)['is_true_positive'].agg(Recall='mean', GT_Count='count'),
        df_pred.groupby('size_bin', observed=False)['is_true_positive'].agg(Precision='mean', Pred_Count='count'),
    ], axis=1)
    table.index.name = f'Size Bin ({unit})'
    table['Bin_mid'] = (bins[:-1] + bins[1:]) / 2
    return table.fillna(0), df_gt, df_pred, bins


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


# ======================================================================================
# DETECTION MODEL: recall, length response and false positives of the recognition pipeline
#
# Turns a sliced theoretical spectrum N_i (expected tracks per bin of etched size, nm) into the
# histogram that OptimusPrimus is expected to report, with uncertainties:
#     recall R_i (true/annotated length) -> length response M_ji -> false positives phi_j (per area)
#     mu_j = sum_i M_ji R_i N_i + phi_j A                       (fp_mode='density', default)
#     mu_j = (sum_i M_ji R_i N_i) / P_j                         (fp_mode='precision')
# Uncertainties: Poisson variance mu_j plus a first-order calibration covariance C_cal.
# The calibration uses the *_detection_logs.json written by OptimusPrimusTraining.evaluate_binned_efficiency
# (see save_detection_logs / load_detection_logs).
# All lengths are in nm; the logged lengths are in um and are converted with `len_scale`.
# ======================================================================================
def _records(log):
    """Converts a validation log into a list of dictionaries.

    Args:
        log (list or pandas.DataFrame): Per-instance records.

    Returns:
        list: The records as dictionaries.
    """
    return log.to_dict('records') if hasattr(log, 'to_dict') else list(log)


def load_detection_logs(path):
    """Reads the validation logs written by `OptimusPrimusTraining.evaluate_binned_efficiency`.

    Args:
        path (str): Path of the *_detection_logs.json file.

    Returns:
        dict: Keys 'gt_log', 'pred_log', 'pair_log', 'val_area_cm2' and 'length_unit'.
    """
    with open(path, 'r') as f:
        logs = json.load(f)
    logs['pair_log'] = [tuple(p) for p in logs['pair_log']]
    return logs


def save_detection_logs(path, gt_log, pred_log, pair_log, val_area_cm2, length_unit='um'):
    """Writes the per-instance validation logs read by `load_detection_logs`.

    Args:
        path (str): Output json file.
        gt_log, pred_log (list): Annotated / reported instances with 'len_um' and 'is_true_positive'.
        pair_log (list): (annotated, reported) lengths of the matched pairs.
        val_area_cm2 (float or None): Total area of the validation images [cm^2] (None without pixel calibration).
        length_unit (str, optional): 'um' or 'px'. Defaults to 'um'.
    """
    payload = {
        'length_unit': length_unit,
        'val_area_cm2': None if val_area_cm2 is None else float(val_area_cm2),
        'gt_log': [{'len_um': float(r['len_um']), 'is_true_positive': bool(r['is_true_positive'])} for r in gt_log],
        'pred_log': [{'len_um': float(r['len_um']), 'is_true_positive': bool(r['is_true_positive'])} for r in pred_log],
        'pair_log': [[float(a), float(b)] for a, b in pair_log],
    }
    with open(path, 'w') as f:
        json.dump(payload, f)


def _length_pair_inliers(gt_len, pred_len, n_mad=5., max_iter=5):
    """Pairs kept for the length-response fit: residuals of a linear bias fit within `n_mad` robust standard
    deviations (1.4826 MAD), iterated. Rejects gross ellipse-fit failures, e.g. on tracks cut by the tile border.

    Returns:
        np.ndarray: Boolean mask of the kept pairs.
    """
    resid = pred_len - gt_len
    keep = np.ones(len(gt_len), bool)
    for _ in range(max_iter):
        b1, b0 = np.polyfit(gt_len[keep], resid[keep], 1)
        r = resid - (b0 + b1 * gt_len)
        scale = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new_keep = np.abs(r) <= n_mad * scale if scale > 0 else keep
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep


def fit_length_response(gt_len, pred_len):
    """Fits the length-measurement response from matched (annotated, reported) pairs.

    The bias is b(L) = b0 + b1 L and the variance sigma^2(L) = s0^2 + (s1 L)^2. Pass inlier pairs (see
    `calibrate_detection_model`, which removes gross outliers first).

    Args:
        gt_len (np.ndarray): Annotated lengths.
        pred_len (np.ndarray): Lengths reported for the same tracks.

    Returns:
        tuple: ((b0, b1), (s0^2, s1^2)).

    Raises:
        ValueError: If fewer than 10 pairs are given.
    """
    gt_len = np.asarray(gt_len, float)
    pred_len = np.asarray(pred_len, float)
    if len(gt_len) < 10:
        raise ValueError("At least 10 matched pairs are needed to fit the length response.")
    resid = pred_len - gt_len
    b1, b0 = np.polyfit(gt_len, resid, 1)
    r2 = (resid - (b0 + b1 * gt_len)) ** 2
    (s0sq, s1sq), _ = _nnls(np.column_stack([np.ones_like(gt_len), gt_len ** 2]), r2)
    return (b0, b1), (s0sq, s1sq)


def calibrate_detection_model(gt_log, pred_log, pair_log, edges_nm, val_area_cm2, len_scale=1.e3, min_count=5):
    """Measures recall, precision, false-positive counts and length response on the annotated validation set.

    Args:
        gt_log (list or pandas.DataFrame): Annotated instances with 'len_um' and 'is_true_positive'.
        pred_log (list or pandas.DataFrame): Reported instances with 'len_um' and 'is_true_positive'.
        pair_log (list): Matched (annotated length, reported length) pairs.
        edges_nm (np.ndarray): Bin edges shared by the spectrum, the validation counts and the measurement [nm].
        val_area_cm2 (float): Total area of the validation images [cm^2].
        len_scale (float, optional): Factor converting the logged lengths to nm. Defaults to 1e3 (um).
        min_count (int, optional): Bins with fewer annotated tracks inherit the recall of the nearest
            populated bin. Defaults to 5.

    Returns:
        dict: Calibration model with the binned counts, the Beta parameters of the recall ('Ra', 'Rb'),
            the length-response parameters ('bias', 'var') fitted on the inlier pairs ('pairs'; the number of
            rejected gross outliers is 'n_pairs_rejected') and the validation area.

    Raises:
        ValueError: If no bin has at least `min_count` annotated tracks.
    """
    edges = np.asarray(edges_nm, float)
    nb = len(edges) - 1
    gt, pr = _records(gt_log), _records(pred_log)
    g_len = np.array([r['len_um'] for r in gt], float) * len_scale
    g_tp = np.array([bool(r['is_true_positive']) for r in gt])
    p_len = np.array([r['len_um'] for r in pr], float) * len_scale
    p_tp = np.array([bool(r['is_true_positive']) for r in pr])
    pairs = np.asarray(pair_log, float).reshape(-1, 2) * len_scale

    def _bins(x):
        idx = np.digitize(x, edges) - 1
        return idx, (idx >= 0) & (idx < nb)

    ig, in_g = _bins(g_len)
    ip, in_p = _bins(p_len)
    G = np.bincount(ig[in_g], minlength=nb)             # annotated tracks per true-length bin
    TPg = np.bincount(ig[in_g & g_tp], minlength=nb)    # ... of which detected
    Q = np.bincount(ip[in_p], minlength=nb)             # detections per reported-length bin
    TPp = np.bincount(ip[in_p & p_tp], minlength=nb)    # ... of which true
    FP = Q - TPp

    Ra, Rb = TPg + 0.5, (G - TPg) + 0.5                 # Beta posterior of the recall (Jeffreys prior)
    ok = np.where(G >= min_count)[0]
    if len(ok) == 0:
        raise ValueError("No bin has enough annotated tracks: use coarser bins.")
    for i in np.where(G < min_count)[0]:
        j = ok[np.argmin(np.abs(ok - i))]
        Ra[i], Rb[i] = Ra[j], Rb[j]

    if len(pairs) < 10:
        raise ValueError("At least 10 matched pairs are needed to fit the length response.")
    inliers = _length_pair_inliers(pairs[:, 0], pairs[:, 1])
    bias, var = fit_length_response(pairs[inliers, 0], pairs[inliers, 1])
    return dict(edges=edges, G=G, TPg=TPg, Q=Q, TPp=TPp, FP=FP, Ra=Ra, Rb=Rb, area=float(val_area_cm2),
                pairs=pairs[inliers], n_pairs_rejected=int((~inliers).sum()), bias=bias, var=var)


def _beta_mean_var(a, b):
    """Mean and variance of a Beta(a, b) distribution."""
    return a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1.))


def binned_recall_precision(model, return_errors=False):
    """Posterior-mean recall (per true-length bin) and precision (per reported-length bin).

    Args:
        model (dict): Output of `calibrate_detection_model`.
        return_errors (bool, optional): If True, the standard deviations are also returned. Defaults to False.

    Returns:
        tuple: (R, P) or (R, P, sigma_R, sigma_P).
    """
    R, varR = _beta_mean_var(model['Ra'], model['Rb'])
    P, varP = _beta_mean_var(model['TPp'] + 0.5, model['FP'] + 0.5)
    return (R, P, np.sqrt(varR), np.sqrt(varP)) if return_errors else (R, P)


def detection_response_matrix(edges, bias, var, n_sub=5):
    """Builds the length-migration matrix M[j, i] = P(reported in bin j | true length in bin i).

    Columns sum to at most one: the remainder migrates outside the histogram range.

    Args:
        edges (np.ndarray): Bin edges [nm].
        bias (tuple): (b0, b1) of the length bias.
        var (tuple): (s0^2, s1^2) of the length variance.
        n_sub (int, optional): Sub-points per true bin used to integrate over the bin. Defaults to 5.

    Returns:
        np.ndarray: Matrix of shape (n_bins, n_bins).
    """
    edges = np.asarray(edges, float)
    nb = len(edges) - 1
    b0, b1 = bias
    s0sq, s1sq = var
    M = np.zeros((nb, nb))
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        L = lo + (np.arange(n_sub) + 0.5) / n_sub * (hi - lo)
        mean = L + b0 + b1 * L
        sig = np.maximum(np.sqrt(s0sq + s1sq * L ** 2), 1.0)
        cdf = _norm.cdf((edges[:, None] - mean[None, :]) / sig[None, :])
        M[:, i] = np.diff(cdf, axis=0).mean(axis=1)
    return M


def _response_shifts(model):
    """Length-response parameters displaced by +1 sigma along each independent direction.

    The directions are the two principal axes of the (b1, b0) fit covariance and a relative error
    sqrt(2 / n_pairs) on each of the two variance coefficients.

    Args:
        model (dict): Output of `calibrate_detection_model`.

    Returns:
        list: Four (bias, var) parameter sets.
    """
    gt, pr = model['pairs'][:, 0], model['pairs'][:, 1]
    _, cov = np.polyfit(gt, pr - gt, 1, cov=True)           # covariance of [b1, b0]
    w, v = np.linalg.eigh(cov)
    (b0, b1), (s0sq, s1sq) = model['bias'], model['var']
    out = []
    for k in range(2):
        d = v[:, k] * np.sqrt(max(w[k], 0.))
        out.append(((b0 + d[1], b1 + d[0]), (s0sq, s1sq)))
    rel = np.sqrt(2. / len(gt))
    out.append(((b0, b1), (s0sq * (1. + rel), s1sq)))
    out.append(((b0, b1), (s0sq, s1sq * (1. + rel))))
    return out


def fold_detection(counts_true, model, area_cm2, fp_mode='density', include_response=True):
    """Folds a sliced spectrum with the detection model.

    Returns the expected reported histogram and the covariance of its calibration uncertainty, propagated to
    first order without sampling. Poisson counting noise (variance mu_j) is not included in the covariance.

    Args:
        counts_true (np.ndarray): Sliced spectrum N_i, expected tracks per bin (e.g. from `slice_spectrum`).
        model (dict): Output of `calibrate_detection_model`.
        area_cm2 (float): Analysed area of the measured sample, which sets the false-positive yield [cm^2].
        fp_mode (str, optional): 'density' takes the false positives from their rate per area (valid for any
            track density); 'precision' divides by the binned precision (same track density as the
            validation set). Defaults to 'density'.
        include_response (bool, optional): Include the uncertainty of the length response. Defaults to True.

    Returns:
        tuple: (mu, C_cal) with the expected counts per bin and their calibration covariance.

    Raises:
        ValueError: If `fp_mode` is not 'density' or 'precision'.
    """
    if fp_mode not in ('density', 'precision'):
        raise ValueError("fp_mode must be 'density' or 'precision'")
    N = np.asarray(counts_true, float)
    R, varR = _beta_mean_var(model['Ra'], model['Rb'])
    P, varP = _beta_mean_var(model['TPp'] + 0.5, model['FP'] + 0.5)
    fpd = (model['FP'] + 0.5) / model['area']               # Gamma posterior mean [per cm^2]
    var_fpd = (model['FP'] + 0.5) / model['area'] ** 2      # Gamma posterior variance

    def _mu(M):
        tp = M @ (R * N)
        return tp + fpd * area_cm2 if fp_mode == 'density' else tp / P

    M = detection_response_matrix(model['edges'], model['bias'], model['var'])
    mu = _mu(M)
    J = M * N[None, :]
    if fp_mode == 'density':
        C = (J * varR[None, :]) @ J.T + np.diag(var_fpd * area_cm2 ** 2)
    else:
        J = J / P[:, None]
        C = (J * varR[None, :]) @ J.T + np.diag(((M @ (R * N)) / P ** 2) ** 2 * varP)
    if include_response:
        for bias, var in _response_shifts(model):
            d = _mu(detection_response_matrix(model['edges'], bias, var)) - mu
            C += np.outer(d, d)
    return mu, C


def merge_bins_auto(mu, min_expected=10.):
    """Greedily merges adjacent bins until each group has at least `min_expected` expected counts.

    Args:
        mu (np.ndarray): Expected counts per bin.
        min_expected (float, optional): Minimum expected counts per merged bin. Defaults to 10.

    Returns:
        list: Lists of bin indices, one per merged bin.
    """
    groups, cur, acc = [], [], 0.
    for j, m in enumerate(mu):
        cur.append(j)
        acc += m
        if acc >= min_expected:
            groups.append(cur)
            cur, acc = [], 0.
    if cur:
        if groups:
            groups[-1] += cur
        else:
            groups.append(cur)
    return groups


def aggregate_bins(observed, mu, C, groups):
    """Merges bins: counts and expectations add, covariances transform as A C A^T.

    Args:
        observed (np.ndarray): Measured counts per bin.
        mu (np.ndarray): Expected counts per bin.
        C (np.ndarray): Covariance of the expected counts.
        groups (list): Bin groups from `merge_bins_auto`.

    Returns:
        tuple: (observed, mu, C) after merging.
    """
    A = np.zeros((len(groups), len(mu)))
    for g, idx in enumerate(groups):
        A[g, idx] = 1.
    return A @ np.asarray(observed, float), A @ mu, A @ C @ A.T


def detection_chi2(observed, mu, C_cal):
    """Chi-square goodness of fit with the full covariance chi2 = (n - mu)^T [diag(mu) + C_cal]^-1 (n - mu).

    The Gaussian approximation needs about 10 expected counts per bin: merge bins first with `merge_bins_auto`.

    Args:
        observed (np.ndarray): Measured counts per (merged) bin.
        mu (np.ndarray): Expected counts per (merged) bin.
        C_cal (np.ndarray): Calibration covariance of the expected counts.

    Returns:
        tuple: (chi2, ndof, p-value).
    """
    d = np.asarray(observed, float) - mu
    chi2 = float(d @ np.linalg.solve(np.diag(mu) + C_cal, d))
    return chi2, len(d), float(_chi2_dist.sf(chi2, len(d)))
# ======================================================================================
# END OF DETECTION MODEL
# ======================================================================================

# ==========================================
# INFERENCE
# ==========================================

class OptimusPrimus:
    """Track detection on z-stacks of image tiles with trained segmentation and classification models."""

    def __init__(self, image_spec, seg_model_spec=None, cls_model_spec=None, image_config=None, tiles_subdir=TILES_SUBDIR,
                 seg_model_config=None, class_model_config=None, parallel=True):
        """
        Args:
            image_spec (str): Image set, i.e. the folder image_folder_path/image_spec/tiles_subdir with the tiles.
            seg_model_spec (str, optional): Segmentation checkpoint name (without .pth). Defaults to the training
                naming scheme for `image_spec`, seg_model_<encoder>_<image_spec>.
            cls_model_spec (str, optional): Classifier checkpoint name (without .pth). Defaults to
                class_model_<encoder>_<image_spec>.
            image_config, seg_model_config, class_model_config (dict, optional): Overrides of the module defaults.
            tiles_subdir (str, optional): Tiles subfolder. Defaults to TILES_SUBDIR.
            parallel (bool, optional): Without a GPU, spread the image-quality analysis over all CPU cores.

        Raises:
            FileNotFoundError: If the tiles folder or a checkpoint does not exist.
        """
        img_cfg = {**IMAGE_CONFIG, **(image_config or {})}
        seg_cfg = {**SEG_MODEL_CONFIG, **(seg_model_config or {})}
        cls_cfg = {**CLASS_MODEL_CONFIG, **(class_model_config or {})}

        self.image_spec = image_spec
        self.image_path = os.path.join(img_cfg['image_folder_path'], image_spec, tiles_subdir)
        self.img_height, self.img_width = img_cfg['img_height'], img_cfg['img_width']
        self.pixel_resolution_um_per_px = img_cfg['pixel_resolution_um_per_px']

        self.seg_model_arc, self.seg_encoder, self.seg_encoder_weights = seg_cfg['model_arc'], seg_cfg['encoder'], seg_cfg['encoder_weights']
        self.cls_model_type = cls_cfg['encoder']
        self.seg_model_th, self.cls_model_th = seg_cfg['threshold'], cls_cfg['threshold']

        self.seg_model_spec = seg_model_spec or make_seg_model_spec(self.seg_encoder, image_spec)
        self.cls_model_spec = cls_model_spec or make_class_model_spec(self.cls_model_type, image_spec)
        self.seg_model_path = os.path.join(seg_cfg['model_folder_path'], self.seg_model_spec + ".pth")
        self.cls_model_path = os.path.join(cls_cfg['model_folder_path'], self.cls_model_spec + ".pth")

        for folder in INFERENCE_OUTPUT_FOLDERS.values():
            os.makedirs(folder, exist_ok=True)
        self.seg_output_path = os.path.join(INFERENCE_OUTPUT_FOLDERS['seg'], f"{image_spec}_{self.seg_model_spec}.csv")
        self.seg_cls_output_path = os.path.join(INFERENCE_OUTPUT_FOLDERS['seg_class'], f"{image_spec}_{self.seg_model_spec}_{self.cls_model_spec}.csv")
        efficiency = efficiency_csv_paths(self.seg_model_spec, self.cls_model_spec, seg_cfg['model_folder_path'], cls_cfg['model_folder_path'])
        self.seg_binned_efficiency_path, self.seg_cls_binned_efficiency_path = efficiency['seg'], efficiency['seg_class']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parallel = parallel
        self.seg_ellipses = self.seg_cls_ellipses = None

        for path, what in ((self.image_path, "Image folder"), (self.seg_model_path, "Segmentation model"), (self.cls_model_path, "Classification model")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{what} not found at '{path}'")

        self.image_groups = self._load_image_groups()
        tile_area_cm2 = self.img_height * self.img_width * (self.pixel_resolution_um_per_px * 1e-4) ** 2 if self.pixel_resolution_um_per_px else None
        self.tot_area_cm2 = len(self.image_groups) * tile_area_cm2 if tile_area_cm2 else None

    def perform_full_inference(self, seg_th=None, cls_th=None, visualize=False, n_top_images=5):
        """Segmentation and segmentation + classification of every z-stack; results are stored in `seg_ellipses` /
        `seg_cls_ellipses` and saved to `seg_output_path` / `seg_cls_output_path`.

        Args:
            seg_th, cls_th (float, optional): Thresholds. Default to the model configurations.
            visualize (bool, optional): Show each image with its final mask.
            n_top_images (int, optional): Sharpest focal planes used per z-stack. Defaults to 5.
        """
        seg_th = self.seg_model_th if seg_th is None else seg_th
        cls_th = self.cls_model_th if cls_th is None else cls_th
        print(f"[OPTIMUS-PRIMUS] {len(self.image_groups)} z-stacks in {self.image_path}")

        stacks = self._select_sharpest(n_top_images)
        seg_model = build_seg_model(self.seg_model_arc, self.seg_encoder, weights_path=self.seg_model_path, device=self.device).eval()
        cls_model = build_class_model(self.cls_model_type, weights_path=self.cls_model_path, device=self.device).eval()
        preprocessing = get_seg_preprocessing(self.seg_encoder, self.seg_encoder_weights, self.img_height, self.img_width)
        class_transform = get_class_transform()

        records = {mode: [] for mode in MODES}
        for group_key, file_list in tqdm(stacks.items(), desc="Processing z-stacks"):
            image, seg_mask = predict_stack_mask(file_list, seg_model, preprocessing, self.device, seg_th)
            cls_mask = create_class_mask(image, seg_mask, cls_model, class_transform, self.device, cls_th)
            name = os.path.basename(file_list[0])
            records['seg'] += self._ellipse_records(seg_mask, name)
            records['seg_class'] += self._ellipse_records(cls_mask, name)
            if visualize:
                show_images([(image, f"Original - {group_key}"), (cls_mask, "Filtered class mask")])

        self.seg_ellipses, self.seg_cls_ellipses = pd.DataFrame(records['seg']), pd.DataFrame(records['seg_class'])
        self.seg_ellipses.to_csv(self.seg_output_path, index=False)
        self.seg_cls_ellipses.to_csv(self.seg_cls_output_path, index=False)
        print(f"[OUTPUT] segmentation: {len(self.seg_ellipses)} tracks -> {self.seg_output_path}")
        print(f"[OUTPUT] segmentation + classification: {len(self.seg_cls_ellipses)} tracks -> {self.seg_cls_output_path}")

    def inference_from_file(self, mode, csv_path=None):
        """Loads saved results of `mode` ('seg' or 'seg_class') from `csv_path` (default: this instance's output path).

        Returns:
            pandas.DataFrame: The loaded tracks.
        """
        _check_mode(mode)
        csv_path = csv_path or (self.seg_output_path if mode == 'seg' else self.seg_cls_output_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at '{csv_path}'")
        df = pd.read_csv(csv_path)
        if mode == 'seg':
            self.seg_ellipses = df
        else:
            self.seg_cls_ellipses = df
        return df

    def get_track_distributions(self, mode, metric='len_um'):
        """Track density over the analysed area (all z-stacks, with or without tracks) and summary statistics of
        `metric`.

        Returns:
            dict: track_density_per_cm2, mean, std, low (25%), median, high (75%).
        """
        _check_mode(mode)
        ellipses = self.seg_ellipses if mode == 'seg' else self.seg_cls_ellipses
        if ellipses is None:
            raise ValueError(f"No '{mode}' results: run perform_full_inference() or inference_from_file() first.")
        density = len(ellipses) / self.tot_area_cm2 if self.tot_area_cm2 else None
        low, median, high = ellipses[metric].quantile([0.25, 0.5, 0.75])
        summary = {"track_density_per_cm2": density, "mean": ellipses[metric].mean(), "std": ellipses[metric].std(),
                   "low": low, "median": median, "high": high}
        area = f"{self.tot_area_cm2:.3f} cm^2" if self.tot_area_cm2 else "unknown area"
        print(f"[{mode}] {len(ellipses)} tracks on {area}; {metric}: mean {summary['mean']:.3f}, std {summary['std']:.3f}, median {median:.3f}")
        return summary

    def detection_logs_path(self, mode='seg_class'):
        """Validation logs of these models, written by OptimusPrimusTraining.evaluate_binned_efficiency."""
        _check_mode(mode)
        return detection_logs_file(self.seg_binned_efficiency_path if mode == 'seg' else self.seg_cls_binned_efficiency_path)

    def _load_image_groups(self):
        """Groups the tiles Frame<n>_Acquisition_<id>_<i>_<j>.png into z-stacks keyed Frame<n>_<i>_<j>."""
        pattern = re.compile(r'Frame(\d+)_Acquisition_(\d+)_(\d+)_(\d+)\.png$')
        groups = {}
        for fpath in sorted(glob.glob(os.path.join(self.image_path, '*.png'))):
            match = pattern.match(os.path.basename(fpath))
            if match:
                n, _, i, j = match.groups()
                groups.setdefault(f"Frame{n}_{i}_{j}", []).append(fpath)
        return groups

    def _select_sharpest(self, n_top_images=5):
        """The `n_top_images` sharpest focal planes of each z-stack, sharpest first."""
        paths = [p for files in self.image_groups.values() for p in files]
        workers = resolve_num_workers(self.parallel)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(tqdm(executor.map(compute_sharpness_score, paths, chunksize=8), total=len(paths), desc="Image quality analysis"))
        else:
            results = [compute_sharpness_score(p) for p in tqdm(paths, desc="Image quality analysis")]
        scores = {}
        for path, score, error in results:
            if error:
                print(error)
            scores[path] = score
        return {key: sorted(files, key=scores.get, reverse=True)[:n_top_images] for key, files in self.image_groups.items()}

    def _ellipse_records(self, mask, image_filename):
        """One record per fitted track of `mask`."""
        records = []
        for track_id, e in enumerate(fit_ellipses(mask), start=1):
            record = {"image_filename": image_filename, "track_id": track_id,
                      "centroid_x_px_ellipse": round(e['center'][0], 1), "centroid_y_px_ellipse": round(e['center'][1], 1),
                      "major_axis_px": round(e['major_px'], 2), "minor_axis_px": round(e['minor_px'], 2),
                      "orientation_deg": round(e['angle_deg'], 2)}
            if self.pixel_resolution_um_per_px is not None:
                record['len_um'] = round(e['major_px'] * self.pixel_resolution_um_per_px, 2)
            records.append(record)
        return records


# ==========================================
# TRAINING AND EVALUATION
# ==========================================

class OptimusPrimusTraining:
    """Training of the segmentation and classification networks and binned evaluation of the pipeline."""

    def __init__(self, train_image_dir, image_spec, train_image=None, training_parameters=None, image_config=None,
                 seg_model_config=None, class_model_config=None, parallel=True):
        """
        Args:
            train_image_dir (str): Folder with the annotated training images; masks in its `mask_subdir`.
            image_spec (str): Name used for the checkpoints (seg_model_<encoder>_<image_spec>, ...).
            train_image, training_parameters, image_config, seg_model_config, class_model_config (dict, optional):
                Overrides of the module defaults. Architectures and encoders come from the model configurations.
            parallel (bool, optional): Without a GPU, load the segmentation data with all CPU cores. With more
                than one GPU, models are wrapped in DataParallel.

        Raises:
            FileNotFoundError: If the image or mask folder does not exist.
        """
        train_cfg = {**TRAIN_IMAGE, **(train_image or {})}
        img_cfg = {**IMAGE_CONFIG, **(image_config or {})}
        seg_cfg = {**SEG_MODEL_CONFIG, **(seg_model_config or {})}
        cls_cfg = {**CLASS_MODEL_CONFIG, **(class_model_config or {})}
        self.params = {**TRAINING_PARAMETERS, **(training_parameters or {})}

        self.image_dir, self.image_spec = train_image_dir, image_spec
        self.mask_dir = os.path.join(train_image_dir, train_cfg['mask_subdir'])
        split_dir = os.path.join(train_image_dir, train_cfg['data_split_subdir'])
        self.split_path = os.path.join(split_dir, train_cfg['split_filename'])
        self.image_extensions, self.mask_extension = train_cfg['image_extensions'], train_cfg['mask_extension']
        self.test_split_ratio, self.val_split_ratio = train_cfg['test_split_ratio'], train_cfg['val_split_ratio']

        # disk-mode classifier patches, next to the training tiles folder
        parent = os.path.dirname(os.path.normpath(train_image_dir))
        self.class_manual_track_dir = os.path.join(parent, 'manual_track')
        self.class_manual_bkg_dir = os.path.join(parent, 'manual_bkg')

        self.image_height, self.image_width = img_cfg['img_height'], img_cfg['img_width']
        self.pixel_resolution_um_per_px = img_cfg['pixel_resolution_um_per_px']

        self.seg_model_arc, self.seg_encoder, self.seg_encoder_weights = seg_cfg['model_arc'], seg_cfg['encoder'], seg_cfg['encoder_weights']
        self.class_model_type, self.class_pretrained = cls_cfg['encoder'], cls_cfg['pretrained']
        self.seg_model_th, self.cls_model_th = seg_cfg['threshold'], cls_cfg['threshold']

        self.seg_best_model_folder = seg_cfg['model_folder_path']
        self.seg_best_model_spec = make_seg_model_spec(self.seg_encoder, image_spec)
        self.seg_best_model_path = os.path.join(self.seg_best_model_folder, f"{self.seg_best_model_spec}.pth")
        self.class_best_model_folder = cls_cfg['model_folder_path']
        self.class_best_model_spec = make_class_model_spec(self.class_model_type, image_spec)
        self.class_best_model_path = os.path.join(self.class_best_model_folder, f"{self.class_best_model_spec}.pth")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parallel = parallel

        for path, what in ((train_image_dir, "Image directory"), (self.mask_dir, "Mask directory")):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{what} not found at '{path}'")
        for folder in (split_dir, self.seg_best_model_folder, self.class_best_model_folder):
            os.makedirs(folder, exist_ok=True)

    # ---------------------------------------------------------------- segmentation

    def perform_seg_training(self):
        """Trains the segmentation network (0.5 Dice + 0.5 BCE on logits, AdamW, ReduceLROnPlateau on the monitored
        validation metric, early stopping after `seg_min_epochs`). The best checkpoint is saved to
        `seg_best_model_path` and the per-epoch metrics next to it."""
        p = self.params
        self._load_split()
        preprocessing = get_seg_preprocessing(self.seg_encoder, self.seg_encoder_weights)
        resize = A.Resize(self.image_height, self.image_width, interpolation=cv2.INTER_LINEAR)
        train_augs = A.Compose([resize, A.HorizontalFlip(p=AUGMENTATION_PARAMETERS['H_FLIP_PROB']),
                                A.VerticalFlip(p=AUGMENTATION_PARAMETERS['V_FLIP_PROB']),
                                A.RandomBrightnessContrast(p=AUGMENTATION_PARAMETERS['BRIGHTNESS_CONTRAST_PROB'])])
        workers, pin = resolve_num_workers(self.parallel), torch.cuda.is_available()
        train_loader = DataLoader(SegmentationDataset(self.train_files, train_augs, preprocessing), batch_size=p['seg_batch_size'],
                                  shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
        val_loader = DataLoader(SegmentationDataset(self.val_files, A.Compose([resize]), preprocessing), batch_size=p['seg_batch_size'],
                                shuffle=False, num_workers=workers, pin_memory=pin)

        model = _parallelize(build_seg_model(self.seg_model_arc, self.seg_encoder, self.seg_encoder_weights, device=self.device))
        dice_loss, bce_loss = smp.losses.DiceLoss(mode="binary", from_logits=True), smp.losses.SoftBCEWithLogitsLoss()

        def loss_fn(pred, target):
            return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

        optimizer = optim.AdamW(model.parameters(), lr=p['seg_learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
        metrics_csv = os.path.join(self.seg_best_model_folder, f"{self.seg_best_model_spec}_training_metrics.csv")
        best, counter, logs = -1.0, 0, []
        try:
            for epoch in range(p['seg_epochs']):
                model.train()
                train_loss = 0.0
                for images, masks in tqdm(train_loader, desc=f"Seg train E{epoch + 1}", leave=False):
                    images, masks = images.to(self.device, dtype=torch.float32), masks.to(self.device, dtype=torch.float32)
                    optimizer.zero_grad()
                    loss = loss_fn(model(images), masks)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_loss, counts = 0.0, np.zeros(3)            # pixel tp, fp, fn over the whole validation set
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(self.device, dtype=torch.float32), masks.to(self.device, dtype=torch.float32)
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
                    best, counter = monitored, 0
                    torch.save(_state_dict(model), self.seg_best_model_path)
                    print(f"Saved segmentation checkpoint: {self.seg_best_model_path}")
                else:
                    counter += 1
                    if counter >= p['seg_patience'] and epoch > p['seg_min_epochs']:
                        print("Early stopping triggered")
                        break
                scheduler.step(monitored)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            pd.DataFrame(logs).to_csv(metrics_csv, index=False)
            print(f"[INFO] Segmentation training finished. Metrics saved at: {metrics_csv}")

    # ---------------------------------------------------------------- classification

    def perform_class_training(self, seg_model_path=None, seg_th=None, keep_patches_in_memory=False, negatives='rims'):
        """Trains the patch classifier on tracks (label 1) and segmentation false positives (label 0).

        Args:
            seg_model_path (str, optional): Segmentation checkpoint used to find the candidates. Defaults to
                `seg_best_model_path`.
            seg_th (float, optional): Segmentation threshold for the candidates. Defaults to the operating point.
            keep_patches_in_memory (bool, optional): Keep the patches in RAM instead of saving them as PNG under
                `class_manual_track_dir` / `class_manual_bkg_dir`. Defaults to False.
            negatives (str, optional): How samples are labelled, 'rims' (default), 'no_rims' or 'unmatched'; see
                `class_training_candidates`.
        """
        p = self.params
        self._load_split()
        train_samples, val_samples = self._create_class_training_dataset(seg_model_path or self.seg_best_model_path, seg_th,
                                                                         keep_patches_in_memory, negatives)
        for split, samples in (('train', train_samples), ('val', val_samples)):
            n_pos = sum(t for _, t in samples)
            if n_pos == 0 or n_pos == len(samples):
                raise ValueError(f"The {split} samples of negatives='{negatives}' have no {'positives' if n_pos == 0 else 'negatives'} "
                                 f"({len(samples)} samples): the classifier cannot be trained. Try a lower segmentation "
                                 f"threshold (more candidates) or another `negatives` mode.")
        train_loader = DataLoader(TrackDataset(train_samples, get_class_transform(train=True)), batch_size=p['class_batch_size'], shuffle=True)
        val_loader = DataLoader(TrackDataset(val_samples, get_class_transform()), batch_size=p['class_batch_size'], shuffle=False)

        model = _parallelize(build_class_model(self.class_model_type, pretrained=self.class_pretrained, device=self.device))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p['class_boost_precision_weight']], device=self.device))
        optimizer = optim.AdamW(model.parameters(), lr=p['class_learning_rate'], weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=p['class_patience'])

        best_dice, logs, eps = -1.0, [], 1e-8          # -1: the first epoch is always saved, even if its Dice is 0
        for epoch in range(p['class_epochs']):
            model.train()
            train_loss = 0.0
            for imgs, labels in tqdm(train_loader, desc=f"Class train E{epoch + 1}", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(model(imgs), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss, tp, fp, fn, tn = 0.0, 0, 0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
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
                best_dice = logs[-1]['val_dice']
                torch.save(_state_dict(model), self.class_best_model_path)
                print(f"Epoch {epoch + 1}: val Dice {best_dice:.4f}, saved classification checkpoint: {self.class_best_model_path}")

        metrics_csv = os.path.join(self.class_best_model_folder, f"{self.class_best_model_spec}_training_metrics.csv")
        pd.DataFrame(logs).to_csv(metrics_csv, index=False)
        print(f"[INFO] Classification training finished. Metrics saved at: {metrics_csv}")

    def _create_class_training_dataset(self, seg_model_path, threshold=None, keep_patches_in_memory=False,
                                       negatives='rims', iou_threshold=0.5):
        """Classifier samples of the train and val splits as 64x64 aligned patches, labelled according to
        `negatives` (see `class_training_candidates`). A summary of the samples is printed and stored in
        `self.class_dataset_summary`.

        Returns:
            tuple: (train_samples, val_samples), lists of (patch array or PNG path, label).
        """
        if negatives not in NEGATIVES_MODES:
            raise ValueError(f"negatives must be one of {NEGATIVES_MODES}, got '{negatives}'")
        threshold = self.seg_model_th if threshold is None else threshold
        seg_model = build_seg_model(self.seg_model_arc, self.seg_encoder, weights_path=seg_model_path, device=self.device).eval()
        preprocessing = get_seg_preprocessing(self.seg_encoder, self.seg_encoder_weights, self.image_height, self.image_width)
        if not keep_patches_in_memory:
            for folder in (self.class_manual_track_dir, self.class_manual_bkg_dir):
                shutil.rmtree(folder, ignore_errors=True)

        self.class_dataset_summary = {}

        def extract(file_list, split):
            samples, counts, n_touching = [], [0, 0], 0
            for img_path, mask_path in tqdm(file_list, desc=f"Patches ({split})", leave=False):
                image, pred_mask = predict_seg_mask(img_path, seg_model, preprocessing, self.device, threshold)
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                for mask, region, target, touches in class_training_candidates(gt_mask, pred_mask, negatives, iou_threshold):
                    n_touching += int(touches and target == 0)
                    patch = get_64x64_centered_patch(image, mask, region)
                    if keep_patches_in_memory:
                        samples.append((patch, target))
                        continue
                    folder = os.path.join(self.class_manual_track_dir if target else self.class_manual_bkg_dir, split)
                    os.makedirs(folder, exist_ok=True)
                    path = os.path.join(folder, f"{'track' if target else 'bkg'}_{counts[target]}.png")
                    cv2.imwrite(path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    samples.append((path, target))
                    counts[target] += 1
            n_pos = sum(t for _, t in samples)
            self.class_dataset_summary[split] = {'positives': n_pos, 'negatives': len(samples) - n_pos, 'negatives_touching_annotation': n_touching}
            print(f"[{split}, negatives='{negatives}'] {n_pos} positives, {len(samples) - n_pos} negatives "
                  f"({n_touching} of them touch an annotation)")
            return samples

        train_samples, val_samples = extract(self.train_files, 'train'), extract(self.val_files, 'val')
        total = len(train_samples) + len(val_samples)
        if keep_patches_in_memory and total > MAX_IN_MEMORY_PATCHES_WARNING:
            print(f"Warning: {total} patches (~{total * 64 * 64 * 3 / 1024 ** 2:.0f} MB) are held in RAM; "
                  f"consider keep_patches_in_memory=False.")
        return train_samples, val_samples

    # ---------------------------------------------------------------- evaluation

    def evaluate_binned_efficiency(self, mode, seg_model_path=None, cls_model_path=None, seg_th=None, cls_th=None,
                                   iou_threshold=0.5, num_bins=10, visualize=False):
        """Instance-level evaluation on the val + test images: Hungarian IoU matching, binned recall and precision
        (csv next to the model) and the per-instance logs for the detection model (json next to the csv).

        Args:
            mode (str): 'seg' (segmentation only) or 'seg_class' (segmentation + classification).
            seg_model_path, cls_model_path (str, optional): Checkpoints. Default to the trained ones.
            seg_th, cls_th (float, optional): Thresholds. Default to the operating points of the model configurations.
            iou_threshold (float, optional): Minimum IoU of a match. Defaults to 0.5.
            num_bins (int, optional): Equal-population length bins of the table. Defaults to 10.
            visualize (bool, optional): Show image, annotation and prediction for each file.

        Returns:
            dict: precision, recall, efficiency_table, df_gt, df_pred, bins, pair_log, val_area_cm2, detection_logs_path.
        """
        _check_mode(mode)
        seg_model_path = seg_model_path or self.seg_best_model_path
        cls_model_path = cls_model_path or self.class_best_model_path
        seg_th = self.seg_model_th if seg_th is None else seg_th
        cls_th = self.cls_model_th if cls_th is None else cls_th

        def spec(path):
            return os.path.splitext(os.path.basename(path))[0]
        efficiency = efficiency_csv_paths(spec(seg_model_path), spec(cls_model_path), self.seg_best_model_folder, self.class_best_model_folder)
        self.seg_binned_efficiency_path, self.seg_cls_binned_efficiency_path = efficiency['seg'], efficiency['seg_class']

        self._load_split()
        seg_model = build_seg_model(self.seg_model_arc, self.seg_encoder, weights_path=seg_model_path, device=self.device).eval()
        cls_model = build_class_model(self.class_model_type, weights_path=cls_model_path, device=self.device) if mode == 'seg_class' else None
        preprocessing = get_seg_preprocessing(self.seg_encoder, self.seg_encoder_weights, self.image_height, self.image_width)
        class_transform = get_class_transform()
        um = self.pixel_resolution_um_per_px

        gt_log, pred_log, pair_log, area_cm2 = [], [], [], 0.0
        for img_path, mask_path in tqdm(self.val_files + self.test_files, desc=f"Evaluating ({mode})"):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image, pred_mask = predict_seg_mask(img_path, seg_model, preprocessing, self.device, seg_th)
            if cls_model is not None:
                pred_mask = create_class_mask(image, pred_mask, cls_model, class_transform, self.device, cls_th)
            g, r, pairs = match_instances_by_iou(gt_mask, pred_mask, iou_threshold, um)
            gt_log += g
            pred_log += r
            pair_log += pairs
            if um is not None:
                area_cm2 += gt_mask.shape[0] * gt_mask.shape[1] * (um * 1e-4) ** 2
            if visualize:
                show_images([(image, "Input"), (gt_mask, "Ground truth"), (pred_mask, mode)], title=os.path.basename(img_path))

        unit = 'um' if um else 'px'
        table, df_gt, df_pred, bins = binned_efficiency_table(gt_log, pred_log, num_bins, unit)
        table.to_csv(efficiency[mode])
        recall, precision = df_gt['is_true_positive'].mean(), df_pred['is_true_positive'].mean()
        n_tp = int(df_gt['is_true_positive'].sum())
        print(f"[{mode}] recall {recall:.3f}, precision {precision:.3f} | TP {n_tp}, "
              f"FP {len(df_pred) - int(df_pred['is_true_positive'].sum())}, FN {len(df_gt) - n_tp}")

        logs_path = detection_logs_file(efficiency[mode])
        save_detection_logs(logs_path, gt_log, pred_log, pair_log, area_cm2 if um is not None else None, unit)
        print(f"[OUTPUT] binned efficiency: {efficiency[mode]} | detection-model logs: {logs_path}")
        return {'precision': precision, 'recall': recall, 'efficiency_table': table, 'df_gt': df_gt, 'df_pred': df_pred,
                'bins': bins, 'pair_log': pair_log, 'val_area_cm2': area_cm2, 'detection_logs_path': logs_path}

    def classifier_operating_curve(self, seg_model_path=None, cls_model_path=None, seg_th=None, thresholds=None, iou_threshold=0.5):
        """Pipeline-level counts, recall and precision of segmentation + classification on the val + test images for a
        sweep of classifier thresholds, with the instance matching of `evaluate_binned_efficiency`. Compare classifiers
        (e.g. trained with different `negatives`) at equal precision instead of at a single threshold.

        Args:
            seg_model_path, cls_model_path (str, optional): Checkpoints. Default to the trained ones.
            seg_th (float, optional): Segmentation threshold. Defaults to the operating point.
            thresholds (array-like, optional): Classifier thresholds. Defaults to 0.05, 0.10, ..., 0.95.
            iou_threshold (float, optional): Minimum IoU of a match. Defaults to 0.5.

        Returns:
            pandas.DataFrame: One row per threshold with TP, FP, FN, recall, precision and f05 (F-beta, beta = 0.5).
        """
        thresholds = np.arange(0.05, 0.96, 0.05) if thresholds is None else np.asarray(thresholds, float)
        seg_th = self.seg_model_th if seg_th is None else seg_th
        self._load_split()
        seg_model = build_seg_model(self.seg_model_arc, self.seg_encoder, weights_path=seg_model_path or self.seg_best_model_path, device=self.device).eval()
        cls_model = build_class_model(self.class_model_type, weights_path=cls_model_path or self.class_best_model_path, device=self.device)
        preprocessing = get_seg_preprocessing(self.seg_encoder, self.seg_encoder_weights, self.image_height, self.image_width)
        class_transform = get_class_transform()

        counts = np.zeros((len(thresholds), 3))                       # TP, FP, FN per threshold
        for img_path, mask_path in tqdm(self.val_files + self.test_files, desc="Operating curve"):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image, pred_mask = predict_seg_mask(img_path, seg_model, preprocessing, self.device, seg_th)
            labeled = label(pred_mask)
            regions = regionprops(labeled)
            probs = np.array(classify_regions(image, pred_mask, regions, cls_model, class_transform, self.device))
            for k, t in enumerate(thresholds):
                kept = np.isin(labeled, [r.label for r, p in zip(regions, probs) if p > t]).astype(np.uint8)
                g, r, _ = match_instances_by_iou(gt_mask, kept, iou_threshold)
                tp = sum(x['is_true_positive'] for x in g)
                counts[k] += [tp, len(r) - sum(x['is_true_positive'] for x in r), len(g) - tp]

        tp, fp, fn = counts.T
        recall, precision = tp / np.maximum(tp + fn, 1), tp / np.maximum(tp + fp, 1)
        f05 = 1.25 * precision * recall / np.maximum(0.25 * precision + recall, 1e-12)
        return pd.DataFrame({'threshold': thresholds, 'TP': tp.astype(int), 'FP': fp.astype(int), 'FN': fn.astype(int),
                             'recall': recall, 'precision': precision, 'f05': f05})

    # ---------------------------------------------------------------- data split

    def _load_split(self):
        """Sets train/val/test (image, mask) pairs from `split_path`, creating and saving the split the first time."""
        if os.path.exists(self.split_path):
            with open(self.split_path, "r") as f:
                splits = json.load(f)
        else:
            pairs = []
            for img_path in sorted(glob.glob(os.path.join(self.image_dir, self.image_extensions))):
                mask_path = os.path.join(self.mask_dir, os.path.splitext(os.path.basename(img_path))[0] + self.mask_extension)
                if os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
            if not pairs:
                raise FileNotFoundError(f"No image-mask pairs found in '{self.image_dir}' / '{self.mask_dir}'")
            test_size, val_size = int(len(pairs) * self.test_split_ratio), int(len(pairs) * self.val_split_ratio)
            train_val, test = train_test_split(pairs, test_size=test_size, random_state=42) if test_size > 0 else (pairs, [])
            train, val = train_test_split(train_val, test_size=val_size, random_state=42) if 0 < val_size < len(train_val) else (train_val, [])
            splits = {"train": train, "val": val, "test": test}
            with open(self.split_path, "w") as f:
                json.dump(splits, f, indent=4)
        self.train_files, self.val_files, self.test_files = splits["train"], splits["val"], splits["test"]