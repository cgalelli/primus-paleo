import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import random
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import rotate
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm as _norm, chi2 as _chi2_dist
from scipy.optimize import nnls as _nnls, linear_sum_assignment
import json

# ==========================================
# CONFIGURATION DICTIONARIES
# ==========================================

IMAGE_CONFIG = {
    'img_height': 1024,
    'img_width': 1024,
    'input_channels_config': 3,
    'pixel_resolution_um_per_px': 0.345,  
    'image_folder_path': "Data/images/",
}

SEG_MODEL_CONFIG = {
    'model_arc': 'MAnet',
    'encoder': 'efficientnet-b7',
    'encoder_weights': 'imagenet',
    'threshold': 0.1,
    'model_folder_path': "Data/models/segmentation/",
}

CLASS_MODEL_CONFIG = {
    'encoder': 'efficientnet_b0',
    'num_classes': 1,
    'model_folder_path': "Data/models/classification/",
    'threshold': 0.55,
}

TRAIN_MODEL_CONFIG ={
    'seg_model_arc': 'MAnet',
    'seg_encoder': 'efficientnet-b7',
    'seg_encoder_weights': 'imagenet',
    'seg_activation': 'sigmoid',
    'num_classes': 1,
    'class_model_type': 'efficientnet_b0',
}

TRAINING_PARAMETERS = {
    'seg_epochs': 200,
    'seg_min_epochs': 100,
    'seg_batch_size': 2,
    'seg_learning_rate': 3e-4,
    'seg_patience': 30,
    'seg_threshold': 0.5,
    'seg_metric_to_monitor': 'Dice/F1',
    'seg_val_beta': 2.0, 
    'seg_loss_alpha': 0.6, 
    'seg_loss_beta': 0.4, 
    'class_boost_precision_weight': 0.7,
    'class_learning_rate': 3e-5,
    'class_epochs': 80,
    'class_batch_size': 64,
    'class_patience': 2,
    'class_threshold': 0.5,
}

TRAIN_IMAGE = {
    'mask_subdir': 'training_masks',
    'data_split_subdir': 'data_split',
    'image_extensions': '*.png',
    'mask_extension': '_mask.png',
    'test_split_ratio': 0.1,
    'val_split_ratio': 0.15,
    'split_filename': 'split_set.json'
}

AUGMENTATION_PARAMETERS = {
    'H_FLIP_PROB': 0.5,
    'V_FLIP_PROB': 0.5,
    'BRIGHTNESS_CONTRAST_PROB': 0.4,
}

# Soft ceiling for `keep_patches_in_memory=True`: above this many total 64x64x3 uint8
# patches (~12 KB each), a warning is printed suggesting disk mode instead, since
# in-memory mode holds every extracted patch for the whole train+val split in RAM
# at once with no eviction.
MAX_IN_MEMORY_PATCHES_WARNING = 50000


TILES_SUBDIR = "tiles"

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def slice_tif_to_png_tiles(images_per_group, input_dir, output_spec=TILES_SUBDIR, tile_size=IMAGE_CONFIG['img_height']):
    """Slices large TIF/TIFF files into smaller uniform square PNG tiles.

    Args:
        images_per_group (int): Number of consecutive images belonging to a single logical group stack.
        input_dir (str): Base filesystem path containing the source high-res TIFF images.
        output_spec (str, optional): Target subfolder name for the extracted PNG tiles. Defaults to TILES_SUBDIR.
        tile_size (int, optional): Spatial edge dimension for square image tiling crops. Defaults to 1024.

    Returns:
        None

    Raises:
        ValueError: If the total number of images found is not divisible by images_per_group.
    """
    output_dir = os.path.join(input_dir, output_spec)
    os.makedirs(output_dir, exist_ok=True)
    
    tif_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])
    
    print(f"Found {len(tif_files)} TIFF images to process with a group size of {images_per_group}.")

    if len(tif_files) % images_per_group != 0:
        raise ValueError(f"The total number of images ({len(tif_files)}) is not a multiple of images_per_group ({images_per_group}).")
    
    file_count = 0
    group_number = 0
    
    for filename in tif_files:
        if file_count % images_per_group == 0:
            group_number += 1
            
        input_filepath = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            img = Image.open(input_filepath)
            width, height = img.size
            x_tiles = width // tile_size
            y_tiles = height // tile_size
            
            if width % tile_size != 0: x_tiles += 1
            if height % tile_size != 0: y_tiles += 1

            print(f"Processing {filename} (Group {group_number}): {x_tiles}x{y_tiles} tiles.")
            
            tile_count = 0
            for i in range(x_tiles):
                for j in range(y_tiles):
                    left = i * tile_size
                    top = j * tile_size
                    right = min(left + tile_size, width)
                    bottom = min(top + tile_size, height)
                    box = (left, top, right, bottom)
                    
                    tile = img.crop(box)
                    if tile.width == tile_size and tile.height == tile_size:
                        tile_filename = f"Frame{group_number}_{base_name}_{i}_{j}.png"
                        output_filepath = os.path.join(output_dir, tile_filename)
                        tile.save(output_filepath, "PNG")
                        tile_count += 1
                    
            print(f"-> Created {tile_count} full-size tiles for {filename}")
            file_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"\n--- Slicing Complete. Total images processed: {file_count} in {group_number} groups. ---")


def convert_xml_to_masks(xml_path, output_dir):
    """Parses CVAT RLE format XML annotations and builds binary ground truth PNG masks.

    Args:
        xml_path (str): Filepath location to the XML annotation dataset.
        output_dir (str): Target directory where the output mask PNG files are stored.

    Returns:
        None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Masks will be saved to: {output_dir}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_tags = root.findall('image')
        
        if not image_tags:
            print("Error: No <image> tags found in the XML file.")
            return

        for image_tag in image_tags:
            image_name = image_tag.get('name')
            img_width = int(image_tag.get('width'))
            img_height = int(image_tag.get('height'))
            final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask_tags = image_tag.findall('mask')
            
            for mask_tag in mask_tags:
                if mask_tag.get('label') != 'Track':
                    continue

                rle_string = mask_tag.get('rle')
                rle_parts = [int(p) for p in rle_string.split(', ')]
                
                values = []
                current_val = 0
                for run_length in rle_parts:
                    values.extend([current_val] * run_length)
                    current_val = 1 - current_val

                mask_height = int(mask_tag.get('height'))
                mask_width = int(mask_tag.get('width'))
                mask_patch = np.array(values, dtype=np.uint8).reshape(mask_height, mask_width)

                left = int(mask_tag.get('left'))
                top = int(mask_tag.get('top'))

                roi = final_mask[top : top + mask_height, left : left + mask_width]
                np.maximum(roi, mask_patch * 255, out=roi)
            
            base_name = os.path.splitext(image_name)[0]
            output_mask_name = f"{base_name}_mask.png"
            output_mask_path = os.path.join(output_dir, output_mask_name)

            cv2.imwrite(output_mask_path, final_mask)

        print(f"\nProcessed {len(image_tags)} images in total.")

    except FileNotFoundError:
        print(f"Error: The file '{xml_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def compute_sharpness_score(fpath):
    """Measures image focus clarity using the variance of the Laplacian operator.

    Args:
        fpath (str): Absolute or relative filepath path of the image target.

    Returns:
        tuple: A tuple containing:
            - fpath (str): The processed image path string.
            - score (float): Calculated variance score (0.0 if unreadable).
            - error_message (str or None): Error description if execution failed, else None.
    """
    try:
        img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return fpath, 0.0, f"[ERROR] Unable to read file: {os.path.basename(fpath)}"

        if np.var(img_gray) == 0:
            score = 0.0
        else:
            score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return fpath, score, None
    except Exception as e:
        return fpath, 0.0, f"{os.path.basename(fpath)}: Unexpected error: {e}"


def get_preprocessing(preprocessing_fn, image_height, image_width):
    """Generates an Albumentations transform pipeline for runtime pre-processing.

    Args:
        preprocessing_fn (callable): Model-specific base pre-processing transform operation.
        image_height (int): Target resizing height boundary.
        image_width (int): Target resizing width boundary.

    Returns:
        A.Compose: The configured Albumentations transformation object mapping.
    """
    _transform = [
        A.Resize(image_height, image_width, interpolation=cv2.INTER_LINEAR),
        A.Lambda(image=preprocessing_fn),
        ToTensorV2(),
    ]
    return A.Compose(_transform)


def get_val_augs():
    """Builds standard evaluation and validation augmentations for classification patches.

    Returns:
        A.Compose: The evaluation transform configuration mapping for 64x64 patches.
    """
    return A.Compose([
        A.Resize(64, 64),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def split_files(folder, train_ratio=0.8, seed=42):
    """Splits filenames randomly into distinct training and validation cohorts.

    Args:
        folder (str): Folder path containing target source files.
        train_ratio (float, optional): Fraction of files allocated to training. Defaults to 0.8.
        seed (int, optional): Random state seed. Defaults to 42.

    Returns:
        tuple: A tuple of two lists containing:
            - train_file_paths (list): List of training paths.
            - validation_file_paths (list): List of validation paths.
    """
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files]
    rnd = random.Random(seed)
    rnd.shuffle(files)
    n_train = int(len(files) * train_ratio)
    return files[:n_train], files[n_train:]


def balance(a, b):
    """Balances the sample counts between two lists by truncating to the smaller size.

    Args:
        a (list): List of sample entries cohort A.
        b (list): List of sample entries cohort B.

    Returns:
        tuple: A balanced pair of lists: (shuffled_a, shuffled_b) matched in length.
    """
    random.shuffle(a)
    random.shuffle(b)
    n = min(len(a), len(b))
    return a[:n], b[:n]


def resolve_num_workers(parallel=True):
    """Resolve the number of parallel CPU workers to use, following the standard
    OptimusPrimus compute policy: look for a GPU first; if one is available, a
    small worker pool is used for CPU-side data prefetching (since the GPU handles
    the heavy computation); if no GPU is available, computation falls back to being
    spread across all available CPU cores unless `parallel` is False, in which case
    a single core (no multiprocessing) is used.

    Args:
        parallel (bool, optional): Whether CPU-side parallel workers should be used
            when no GPU is available. Defaults to True.

    Returns:
        int: Number of worker processes/threads to use.
    """
    cpu_count = os.cpu_count() or 1
    if torch.cuda.is_available():
        return min(4, cpu_count // 2) if cpu_count > 1 else 0
    return cpu_count if parallel else 0


def extract_instances(binary_mask):
    """Segments connected components from a binary mask and extracts geometrical properties.

    Args:
        binary_mask (np.ndarray): Single channel binary mask array (0 or 255 / 1).

    Returns:
        list: Collection of descriptive feature dictionaries for each extracted component instance.
    """
    contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    instances = []

    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) < 10 or len(contour) < 5:
            continue
        
        ellipse = cv2.fitEllipse(contour)
        ((_, _), (axis_a_px, axis_b_px), _) = ellipse
        major_axis_len_px = max(axis_a_px, axis_b_px)
        
        size_metric = major_axis_len_px
        pixel_resolution_um_per_px = IMAGE_CONFIG['pixel_resolution_um_per_px']
        if pixel_resolution_um_per_px is not None:
            size_metric = major_axis_len_px * pixel_resolution_um_per_px

        instance_mask = np.zeros_like(binary_mask)
        cv2.drawContours(instance_mask, [contour], -1, 1, thickness=cv2.FILLED)

        props = regionprops(instance_mask.astype(np.uint8))[0]

        instances.append({
            "id": idx,
            "size": size_metric,
            "mask": instance_mask,
            "centroid": props.centroid,
            "bbox": props.bbox,
            "orientation": props.orientation,
            "contour": contour,
            "props": props,
            "matched": False
        })

    return instances


def get_64x64_centered_patch(img, mask, region):
    """Extracts and aligns a 64x64 pixel normalized image patch around an object region's centroid.

    Args:
        img (np.ndarray): Full scale source RGB image array.
        mask (np.ndarray): Full scale binary object mask layer.
        region (RegionProperties): Structural region properties extracted from scikit-image.

    Returns:
        np.ndarray: Centered, rotated, and normalized 64x64x3 RGB image patch array.
    """
    cy, cx = region.centroid
    minr, minc, maxr, maxc = region.bbox
    diag = int(np.sqrt((maxr-minr)**2 + (maxc-minc)**2)) + 10
    
    p_img = np.pad(img, ((diag, diag), (diag, diag), (0, 0)), mode='constant')
    p_mask = np.pad(mask, ((diag, diag), (diag, diag)), mode='constant')
        
    p_cy, p_cx = int(cy + diag), int(cx + diag) 
        
    square_img = p_img[p_cy-diag : p_cy+diag, p_cx-diag : p_cx+diag]
    square_mask = p_mask[p_cy-diag : p_cy+diag, p_cx-diag : p_cx+diag]

    current_angle = np.rad2deg(-region.orientation)
    vertical_angle = current_angle - 90
        
    rot_img = rotate(square_img, vertical_angle, resize=False, order=1, preserve_range=True).astype(np.uint8)
    rot_mask = rotate(square_mask, vertical_angle, resize=False, order=0, preserve_range=True)

    rot_labeled = label(rot_mask > 0.5)
    h_rot, w_rot = rot_mask.shape
    center_label = rot_labeled[h_rot // 2, w_rot // 2]
        
    if center_label == 0:
        obj_coords = np.argwhere(rot_labeled > 0)
        if obj_coords.size > 0:
            center_pt = np.array([h_rot // 2, w_rot // 2])
            distances = np.linalg.norm(obj_coords - center_pt, axis=1)
            closest_idx = np.argmin(distances)
            center_label = rot_labeled[obj_coords[closest_idx][0], obj_coords[closest_idx][1]]
    
    final_single_mask = (rot_labeled == center_label).astype(np.uint8)
    contours, _ = cv2.findContours(final_single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(contours[0])
    tight_obj_rgb = rot_img[y:y+h, x:x+w]
    
    h, w = tight_obj_rgb.shape[:2]
    if h > 64 or w > 64:
        scale = min(64/h, 64/w)
        tight_obj_rgb = cv2.resize(tight_obj_rgb, (int(w*scale), int(h*scale)))
        h, w = tight_obj_rgb.shape[:2]

    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    start_y, start_x = (64 - h) // 2, (64 - w) // 2
    canvas[start_y:start_y+h, start_x:start_x+w] = tight_obj_rgb
    return canvas


def evaluate_patch(patch, model, transform, device, threshold=None):
    """Evaluates a single cropped image patch through the classification model filter.

    Args:
        patch (np.ndarray): The 64x64 pixel patch array configuration.
        model (nn.Module): The classification network PyTorch module instance.
        transform (A.Compose): Pre-processing normalization transformations mapper.
        device (str): Inference device runtime framework target (e.g. 'cuda' or 'cpu').
        threshold (float, optional): Classification activation threshold boundary. Defaults to None.

    Returns:
        bool: True if the model classifies the input patch as a real feature track, False otherwise.
    """
    if threshold is None:
        threshold = 0.5
    
    model.eval()
    x = transform(image=patch)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.sigmoid(output)
        track_prob = probs[0, 0].item()

    return track_prob > threshold


def create_class_mask(image, mask, model, transform, device, threshold=None):
    """Applies the secondary patch classifier over instances found inside the segmentation mask.

    Args:
        image (np.ndarray): Base full scale RGB frame canvas.
        mask (np.ndarray): Predicted binary segmentation layer array.
        model (nn.Module): Pre-trained classification filtering network module.
        transform (A.Compose): Normalization scaling transform mapping configurations.
        device (str): Processing system platform target string.
        threshold (float, optional): Classification validation pass boundary filter. Defaults to None.

    Returns:
        np.ndarray: Cleaned binary mask containing only validated true-positive tracks.
    """
    final_confirmed_mask = np.zeros_like(mask)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    for region in regions:
        patch = get_64x64_centered_patch(image, mask, region)
        is_valid = evaluate_patch(patch, model, transform, device, threshold=threshold)
            
        if is_valid:
            final_confirmed_mask[labeled_mask == region.label] = 1
    return final_confirmed_mask

# ==========================================
# CORE PIPELINE CLASSES
# ==========================================

# ======================================================================================
# DETECTION MODEL: recall, length response and false positives of the recognition pipeline
#
# Turns a sliced theoretical spectrum N_i (expected tracks per bin of etched size, nm) into the
# histogram that OptimusPrimus is expected to report, with uncertainties:
#     recall R_i (true/annotated length) -> length response M_ji -> false positives phi_j (per area)
#     mu_j = sum_i M_ji R_i N_i + phi_j A                       (fp_mode='density', default)
#     mu_j = (sum_i M_ji R_i N_i) / P_j                         (fp_mode='precision')
# Uncertainties: Poisson variance mu_j plus a first-order calibration covariance C_cal.
# The calibration uses the *_detection_logs.json written by OptimusPrimusTraining.evaluate_binned_efficiency.
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


def fit_length_response(gt_len, pred_len):
    """Fits the length-measurement response from matched (annotated, reported) pairs.

    The bias is b(L) = b0 + b1 L and the variance sigma^2(L) = s0^2 + (s1 L)^2.

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
            the length-response parameters ('bias', 'var') and the validation area.

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

    bias, var = fit_length_response(pairs[:, 0], pairs[:, 1])
    return dict(edges=edges, G=G, TPg=TPg, Q=Q, TPp=TPp, FP=FP, Ra=Ra, Rb=Rb,
                area=float(val_area_cm2), pairs=pairs, bias=bias, var=var)


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


class OptimusPrimus:
    """Production runtime inference management engine for the track identification architecture."""
    
    def __init__(self, image_spec, seg_model_spec, cls_model_spec, image_config=None, tiles_subdir=TILES_SUBDIR, seg_model_config=None, class_model_config=None, parallel=True):
        """Initializes structural paths, parameters, and system environments for production deployment.

        Args:
            image_spec (str): Key identifier describing the processing image group stack dataset.
            seg_model_spec (str): Name string of the saved segmentation weights file checkpoint.
            cls_model_spec (str): Name string of the saved classification weights file checkpoint.
            image_config (dict, optional): Custom execution configurations overrides for images. Defaults to None.
            tiles_subdir (str, optional): Sub-path folder containing PNG tiled arrays. Defaults to TILES_SUBDIR.
            seg_model_config (dict, optional): Custom configurations overrides for the segmentation module. Defaults to None.
            class_model_config (dict, optional): Custom configurations overrides for the classification module. Defaults to None.
            parallel (bool, optional): Standard compute policy toggle. A CUDA GPU is always used when
                available; when no GPU is available, CPU-bound work (e.g. image quality analysis) is
                spread across all available CPU cores unless `parallel` is set to False, in which case
                it runs on a single core. Defaults to True.

        Raises:
            FileNotFoundError: If target image paths or designated check-point models do not exist on disk.
        """
        _image_config = IMAGE_CONFIG.copy()
        if image_config: _image_config.update(image_config)
        
        _seg_model_config = SEG_MODEL_CONFIG.copy()
        if seg_model_config: _seg_model_config.update(seg_model_config)
            
        _cls_model_config = CLASS_MODEL_CONFIG.copy()
        if class_model_config: _cls_model_config.update(class_model_config)
        
        self.image_folder_path = _image_config['image_folder_path']
        self.image_spec = image_spec 
        self.sub_dir = tiles_subdir
        self.image_path = os.path.join(self.image_folder_path, self.image_spec, self.sub_dir)
        
        self.seg_model_folder_path = _seg_model_config['model_folder_path']
        self.seg_model_spec = seg_model_spec
        self.model_ext = ".pth"
        self.seg_model_path = os.path.join(self.seg_model_folder_path, self.seg_model_spec + self.model_ext)

        self.cls_model_folder_path = _cls_model_config['model_folder_path']
        self.cls_model_spec = cls_model_spec
        self.cls_model_path = os.path.join(self.cls_model_folder_path, self.cls_model_spec + self.model_ext)
        
        self.seg_output_folder_path = "Data/inference_results/segmentation/"
        os.makedirs(self.seg_output_folder_path, exist_ok=True)
        self.seg_output_spec = image_spec + "_" + seg_model_spec + ".csv"
        self.seg_output_path = os.path.join(self.seg_output_folder_path, self.seg_output_spec)
        
        self.seg_cls_output_folder_path = "Data/inference_results/seg_class/"
        os.makedirs(self.seg_cls_output_folder_path, exist_ok=True)
        self.seg_cls_output_spec = image_spec + "_" + seg_model_spec + "_" + cls_model_spec  + ".csv"
        self.seg_cls_output_path = os.path.join(self.seg_cls_output_folder_path, self.seg_cls_output_spec)
        
        self.seg_efficiency_input_spec = "binned_efficiency_" + self.seg_model_spec + ".csv"
        self.seg_binned_efficiency_path = os.path.join(self.seg_model_folder_path, self.seg_efficiency_input_spec)
        
        self.seg_cls_efficiency_input_spec = "binned_efficiency_" + self.seg_model_spec + "_" + self.cls_model_spec  + ".csv"
        self.seg_cls_binned_efficiency_path = os.path.join(self.cls_model_folder_path, self.seg_cls_efficiency_input_spec)
        
        self.seg_model_arc = _seg_model_config['model_arc']
        self.seg_encoder = _seg_model_config['encoder']
        self.seg_encoder_weights = _seg_model_config['encoder_weights']
        self.seg_model_th = _seg_model_config['threshold']
        
        self.cls_model_encoder = _cls_model_config['encoder']
        self.cls_model_th = _cls_model_config['threshold']
        self.cls_num_classes = _cls_model_config['num_classes']
        
        self.img_height = _image_config['img_height']
        self.img_width = _image_config['img_width']
        self.input_channels_config = _image_config['input_channels_config']
        self.pixel_resolution_um_per_px = _image_config['pixel_resolution_um_per_px']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parallel = parallel
        self.seg_efficiency_table = None
        self.seg_cls_efficiency_table = None

        if not os.path.isdir(self.image_path):
            raise FileNotFoundError(f"ERROR: Image folder not found at '{self.image_path}'")
        if not os.path.exists(self.seg_model_path):
            raise FileNotFoundError(f"ERROR: Segmentation model file not found at '{self.seg_model_path}'")
        if not os.path.exists(self.cls_model_path):
            raise FileNotFoundError(f"ERROR: Classification model file not found at '{self.cls_model_path}'")

    def perform_full_inference(self, seg_th=None, cls_th=None, visualize=False):
        """Executes the complete segmentation + classification inference stack over loaded Z-stacks.

        Args:
            seg_th (float, optional): Customized probability threshold for segmentation. Defaults to None.
            cls_th (float, optional): Customized probability threshold for classification filtering. Defaults to None.
            visualize (bool, optional): Activates interactive on-screen matplotlib verification windows. Defaults to False.

        Returns:
            None
        """
        if seg_th is None: seg_th = self.seg_model_th
        if cls_th is None: cls_th = self.cls_model_th
        
        print("\n[OPTIMUS-PRIMUS | SEGMENTATION + CLASSIFICATION PIPELINE] START\n")
        print("[STEP 1/7] Loading image groups...")
        image_groups = self._load_image_groups()
        
        print("[STEP 2/7] Performing image quality filtering...")
        top_image_groups = self._image_quality_analysis(image_groups)
        
        print("[STEP 3/7] Loading segmentation model...")
        inference_model = self._load_seg_model()
        
        print("[STEP 4/7] Loading classification model...")
        cls_model = self._load_cls_model()

        print("[STEP 5/7] Initializing preprocessing pipeline...")
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.seg_encoder, self.seg_encoder_weights)
        preprocessing = get_preprocessing(preprocessing_fn, self.img_height, self.img_width)
        class_transform = get_val_augs()

        seg_ellipse_analysis_results = []
        seg_cls_ellipse_analysis_results = []
    
        print("[STEP 6/7] Running segmentation + classification inference...")
        for group_key, file_list in tqdm(top_image_groups.items(), desc="Processing Z-stacks"):
            original_vis_image, aggregated_mask = self._create_segmentation_mask(file_list, preprocessing, inference_model, threshold=seg_th)
            final_confirmed_mask = create_class_mask(original_vis_image, aggregated_mask, cls_model, class_transform, self.device, threshold=cls_th)  
            
            seg_ellipse_analysis_results = self._analyze_mask('seg', aggregated_mask, seg_ellipse_analysis_results, file_list, group_key)
            seg_cls_ellipse_analysis_results = self._analyze_mask('seg_class', final_confirmed_mask, seg_cls_ellipse_analysis_results, file_list, group_key)
            
            if visualize:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Original - {group_key}")
                plt.imshow(original_vis_image)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title("Filtered Class Mask")
                plt.imshow(final_confirmed_mask, cmap='gray')
                plt.axis('off')
                plt.show()
    
        print("[STEP 7/7] Saving results...")
        df_seg_ellipse_results = pd.DataFrame(seg_ellipse_analysis_results)
        df_seg_cls_ellipse_results = pd.DataFrame(seg_cls_ellipse_analysis_results)
        df_seg_ellipse_results.to_csv(self.seg_output_path, index=False)
        df_seg_cls_ellipse_results.to_csv(self.seg_cls_output_path, index=False)

        self.seg_ellipses = df_seg_ellipse_results
        self.seg_cls_ellipses = df_seg_cls_ellipse_results
        
        print("\n[OPTIMUS-PRIMUS | SEGMENTATION + CLASSIFICATION PIPELINE] DONE")
        print(f"[OUTPUT] segmentation CSV saved at: {self.seg_output_path}")
        print(f"[OUTPUT] segmentation & classification CSV saved at: {self.seg_cls_output_path}")

    def inference_from_file(self, csv_path=None):
        """Loads extraction ellipse analysis records directly from an existing CSV results file.

        Args:
            csv_path (str, optional): Target filesystem source location path. Defaults to None.

        Returns:
            None

        Raises:
            FileNotFoundError: If the designated CSV file path does not exist on disk.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ERROR: CSV file not found at '{csv_path}'")
        
        if 'segmentation' in csv_path:
            df_ellipse_results = pd.read_csv(csv_path)
            self.seg_ellipses = df_ellipse_results
        elif 'seg_class' in csv_path:
            df_ellipse_results = pd.read_csv(csv_path)
            self.seg_cls_ellipses = df_ellipse_results
        else:
            print("csv_path isn't in a folder named 'segmentation' or 'seg_class'")

    def get_track_distributions(self, mode, metric='len_um'):
        """Computes statistical metrics and track spatial density metrics for processed datasets.

        Args:
            mode (str): Evaluation path mode selection ('seg' or 'seg_class').
            metric (str, optional): DataFrame metric key target for statistical calculation. Defaults to 'len_um'.

        Returns:
            dict: Structured statistical summary distribution attributes mapping.

        Raises:
            ValueError: If target internal evaluation frames are not loaded or if mode selection is invalid.
        """
        print(f"\n[OPTIMUS-PRIMUS | DISTRIBUTION ANALYSIS] MODE = {mode}")
        print(f"[STEP 1] Loading ellipse dataset...")
        
        if mode == 'seg':
            if not hasattr(self, 'seg_ellipses'):
                raise ValueError(f"No 'seg_ellipses' data found. Please run perform_full_inference() first or inference_from_file().")
            ellipses = self.seg_ellipses
        elif mode == 'seg_class':
            if not hasattr(self, 'seg_cls_ellipses'):
                raise ValueError("No seg_cls_ellipse data found. Please run perform_full_inference() first or inference_from_file().")
            ellipses = self.seg_cls_ellipses
        else:
            raise ValueError("mode need to be 'seg' or 'seg_class'")
        
        print(f"[STEP 2] Dataset loaded: {len(ellipses)} tracks found")
        print("[STEP 3] Computing dataset area and density...")
        
        self.img_area_cm2 = self.img_height * self.img_width * (self.pixel_resolution_um_per_px * 1e-4)**2
        self.tot_area_cm2 = len(np.unique(ellipses['image_filename'])) * self.img_area_cm2
        track_density = len(ellipses) / self.tot_area_cm2

        print(f"[INFO] Total area: {self.tot_area_cm2:.3f} cm²")
        print(f"[INFO] Track density: {track_density:.3f} cm⁻²")
        print(f"[STEP 4] Computing statistics on '{metric}'...")

        mean_value = ellipses[metric].mean()
        std_value = ellipses[metric].std()
        low, median, high = ellipses[metric].quantile([0.25, 0.5, 0.75])
        
        print(f"[INFO] Mean = {mean_value:.3f}, Std = {std_value:.3f}")
        print(f"[INFO] Median = {median:.3f}")

        distribution_summary = {
            "track_density_per_cm2": track_density,
            "mean": mean_value,
            "std": std_value,
            "low": low,
            "median": median,
            "high": high
        }
        
        print("[DONE] Distribution analysis completed\n")
        return distribution_summary

    def efficiency_distribution_from_file(self, csv_path=None):
        """Loads predefined tracking efficiency evaluation tables directly from storage.

        Args:
            csv_path (str, optional): CSV source filesystem location. Defaults to None.

        Returns:
            None

        Raises:
            FileNotFoundError: If the designated calibration file is not found on disk.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ERROR: CSV file not found at '{csv_path}'")
        
        if 'segmentation' in csv_path:
            self.seg_efficiency_table = pd.read_csv(csv_path, index_col=0)
            print("Loading efficiency table from CSV and storing it in self.seg_efficiency_table.")
        elif 'seg_class' in csv_path:
            self.seg_cls_efficiency_table = pd.read_csv(csv_path, index_col=0)
            print("Loading efficiency table from CSV and storing it in self.seg_cls_efficiency_table.")

    def _load_image_groups(self):
        """Scans the image directory and groups multi-focus sequential structures via regex parsing.

        Returns:
            defaultdict: Map tracking frame acquisition coordinate blocks to raw tile paths.

        Raises:
            FileNotFoundError: If the base tile processing directory structure does not exist.
        """
        if not os.path.isdir(self.image_path):
            raise FileNotFoundError(f"ERROR: Image folder not found at '{self.image_path}'")

        image_files = sorted(glob.glob(os.path.join(self.image_path, '*.png')))
        file_pattern = re.compile(r'Frame(\d+)_Acquisition_(\d+)_(\d+)_(\d+).png')
        image_groups = defaultdict(list)

        for fpath in image_files:
            fname = os.path.basename(fpath)
            match = file_pattern.match(fname)
            if match:
                n, image_id, top, left = match.groups()
                group_key = f"Frame{n}_{top}_{left}"
                image_groups[group_key].append(fpath)

        img_area_cm2 = self.img_height * self.img_width * (self.pixel_resolution_um_per_px * 1e-4)**2
        self.tot_area_cm2 = len(image_groups) * img_area_cm2
        return image_groups
    
    def _image_quality_analysis(self, image_groups, max_workers=None, n_top_images=5):
        """Filters spatial group cohorts down to top focus stacks using concurrent workers.

        Args:
            image_groups (dict): Structured coordinate maps holding candidate file groups.
            max_workers (int, optional): Execution ceiling constraint limits for concurrent processes.
                Defaults to None, in which case it is resolved from the standard compute policy
                (`resolve_num_workers(self.parallel)`): a GPU present -> a small worker pool; no GPU
                present -> all CPU cores, unless `self.parallel` is False, in which case a single
                worker (sequential execution) is used.
            n_top_images (int, optional): Target selection quantity constraint per unique stack. Defaults to 5.

        Returns:
            defaultdict: Pruned image collection retaining high-clarity focal structures.
        """
        if max_workers is None:
            max_workers = resolve_num_workers(self.parallel) or 1

        group_scores = defaultdict(list)
        all_tasks = [(group_key, fpath) for group_key, file_paths in image_groups.items() for fpath in file_paths]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(compute_sharpness_score, fpath): group_key for group_key, fpath in all_tasks}

            for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Image quality analysis"):
                group_key = future_to_key[future]
                fpath, score, error_msg = future.result()
                if error_msg: print(error_msg)
                group_scores[group_key].append((fpath, score))

        top_images_groups = defaultdict(list)
        for group_key, scores_list in group_scores.items():
            if not scores_list: continue
            sorted_list = sorted(scores_list, key=lambda item: item[1], reverse=True)
            if len(sorted_list) > n_top_images: sorted_list = sorted_list[:n_top_images]
            for rank, (fpath, score) in enumerate(sorted_list, start=1):
                top_images_groups[group_key].append(fpath)

        return top_images_groups

    def _load_seg_model(self):
        """Constructs and initializes the SMP segmentation network from storage weights.

        Returns:
            nn.Module: Loaded and deployed segmentation network wrapper ready for evaluation.
        """
        seg_model = smp.create_model(
            arch=self.seg_model_arc, encoder_name=self.seg_encoder, encoder_weights=None,
            in_channels=self.input_channels_config, classes=1, activation=None, 
        )
        seg_model.load_state_dict(torch.load(self.seg_model_path, map_location=self.device, weights_only=True))
        seg_model.to(self.device)
        seg_model.eval()
        return seg_model
        
    def _load_cls_model(self):
        """Constructs and initializes the custom classification feature filter network.

        Returns:
            nn.Module: Positioned classification network workspace model loaded on target hardware.
        """
        cls_model = timm.create_model(self.cls_model_encoder, pretrained=False, num_classes=self.cls_num_classes)
        cls_model.conv_stem.stride = (1, 1)
        cls_model.blocks[1][0].conv_dw.stride = (1, 1)
        in_features = cls_model.classifier.in_features
        cls_model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))

        state_dict = torch.load(self.cls_model_path, map_location=self.device, weights_only=True)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        cls_model.load_state_dict(state_dict)
        cls_model.to(self.device)
        return cls_model

    def _compute_seg_masks(self, image_path, model, preprocessing, threshold=None):
        """Processes a single file path tracking target to compile raw prediction arrays.

        Args:
            image_path (str): File location target of the PNG crop frame.
            model (nn.Module): Active segmentation network instance.
            preprocessing (A.Compose): Input formatting transformation layer mappings.
            threshold (float, optional): Custom execution threshold limit value. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - image (np.ndarray): Original transformed image tensor canvas.
                - pred_mask (np.ndarray): Extracted binary segmentation mask array.
        """
        if threshold is None: threshold = self.seg_model_th
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        sample = preprocessing(image=image)
        preprocessed = sample['image'].unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(preprocessed)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > threshold).cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask.squeeze()
        return image, pred_mask
    
    def _create_segmentation_mask(self, file_list, preprocessing, seg_model, threshold=None):
        """Aggregates maximum pixel probability across focus layers to compile tracking maps.

        Args:
            file_list (list): Collection of image slice paths comprising a focus stack.
            preprocessing (A.Compose): Model feature normalization transform pipelines.
            seg_model (nn.Module): Running PyTorch segmentation processor module.
            threshold (float, optional): Operational detection probability limit. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - original_vis_image (np.ndarray): First-frame baseline canvas reference image.
                - aggregated_mask (np.ndarray): Merged maximum intensity projection prediction mask.
        """
        group_masks = []
        original_vis_image = None
        for idx, image_path in enumerate(file_list):
            image, pred_mask = self._compute_seg_masks(image_path, seg_model, preprocessing, threshold=threshold)
            if idx == 0: original_vis_image = image.copy()
            group_masks.append(pred_mask)
        aggregated_mask = np.max(np.stack(group_masks, axis=0), axis=0).astype(np.uint8)
        return original_vis_image, aggregated_mask
    
    def _analyze_mask(self, mode, mask, analysis_result, file_list, group_key):
        """Fits structural ellipses over tracking features and records parameters to results.

        Args:
            mode (str): Pipeline operational branch type description ('seg' or 'seg_class').
            mask (np.ndarray): Binary segmentation matrix array containing track shapes.
            analysis_result (list): Cumulative list storage array container for data frames.
            file_list (list): Tracking file source lists identifying current locations.
            group_key (str): Spatial stack group coordinate sequence naming tag.

        Returns:
            list: Aggregated parameter track feature record collection update tracking.
        """
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        track_id_counter = 1
                    
        for contour in contours:
            if cv2.contourArea(contour) < 10 or len(contour) < 5: continue
            ellipse = cv2.fitEllipse(contour)
            ((center_x, center_y), (axis_a_px, axis_b_px), orientation_deg) = ellipse
            major_axis_len_px = max(axis_a_px, axis_b_px)
            minor_axis_len_px = min(axis_a_px, axis_b_px)
                        
            track_props = {
                "image_filename": os.path.basename(file_list[0]), 
                "track_id": track_id_counter,
                "centroid_x_px_ellipse": round(center_x, 1), 
                "centroid_y_px_ellipse": round(center_y, 1),
                "major_axis_px": round(major_axis_len_px, 2),
                "minor_axis_px": round(minor_axis_len_px, 2),
                "orientation_deg": round(orientation_deg, 2),
            }
                        
            if self.pixel_resolution_um_per_px is not None:
                track_props['len_um'] = round(major_axis_len_px * self.pixel_resolution_um_per_px, 2)
        
            analysis_result.append(track_props)
            track_id_counter += 1
        
        return analysis_result


class OptimusPrimusTraining:
    """Consolidated model training optimization, validation, and benchmarking manager."""

    class TrackDataset(Dataset):
        """Custom dataset targeting standard sub-cropped micro structure patch arrays."""
        
        def __init__(self, samples, transform=None):
            """Initializes the dataset with specific patches and target matching labels.

            Args:
                samples (list): Collection list containing matching tuples of (patch, label_int), where
                    patch is either an in-memory patch_array (np.ndarray) or a file path (str) to a
                    saved patch on disk, depending on how the dataset was generated.
                transform (A.Compose, optional): Albumentations augmentations pipelines layer. Defaults to None.
            """
            self.samples = samples  
            self.transform = transform

        def __len__(self):
            """Returns total sample record counts.

            Returns:
                int: Total length number value of entries in samples list.
            """
            return len(self.samples)

        def __getitem__(self, idx):
            """Retrieves a single augmented item matching target indexing metrics.

            Args:
                idx (int): Absolute numeric matrix lookup index coordinate.

            Returns:
                tuple: A tuple containing:
                    - patch_tensor (torch.Tensor): Processed float matrix patch tensor.
                    - label (torch.Tensor): Standard single-class float classification binary targets.
            """
            patch, label = self.samples[idx]
            if isinstance(patch, str):
                # Disk mode: sample is a file path, load it from disk.
                patch = cv2.imread(patch)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=patch)
                patch_tensor = augmented['image']
            else:
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
            return patch_tensor, torch.tensor([label], dtype=torch.float32)

    class SegmentationDataset(Dataset):
        """PyTorch wrapper providing optimized access to multi-channel semantic mask objects."""
        
        def __init__(self, file_pairs, augmentations=None, preprocessing=None, input_channels=3):
            """Initializes semantic target input sequences.

            Args:
                file_pairs (list): Collection list of coordinate strings matching (image_path, mask_path).
                augmentations (A.Compose, optional): Geometric distortion transform configurations. Defaults to None.
                preprocessing (A.Compose, optional): Encoder pixel scaling pipelines normalization. Defaults to None.
                input_channels (int, optional): Spatial dimensions channel counting ceiling profiles. Defaults to 3.
            """
            self.image_files = [pair[0] for pair in file_pairs]
            self.mask_files = [pair[1] for pair in file_pairs]
            self.augmentations = augmentations
            self.preprocessing = preprocessing
            self.input_channels = input_channels 

        def __len__(self):
            """Returns sequence allocation caps.

            Returns:
                int: Count limits metric total number of element assets.
            """
            return len(self.image_files)

        def __getitem__(self, idx):
            """Funnels data assets from storage to runtime transformation matrices.

            Args:
                idx (int): Data locator array index target mapping.

            Returns:
                tuple: A tuple containing:
                    - image_tensor (torch.Tensor): Scaled image tensor block array.
                    - mask_tensor (torch.Tensor): Ground truth semantic label layout space tensor.
            """
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            if self.input_channels == 1:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else: 
                image = cv2.imread(img_path, cv2.IMREAD_COLOR) 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)

            if self.augmentations:
                augmented = self.augmentations(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask'] 

            if self.preprocessing:
                preprocessed = self.preprocessing(image=image)
                image_tensor = preprocessed['image'] 
                if mask.ndim == 2: mask_tensor = torch.from_numpy(mask).unsqueeze(0) 
                elif mask.ndim == 3: mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1))
                mask_tensor = mask_tensor.float()
            else:
                if image.ndim == 3: image = image.transpose(2, 0, 1) 
                elif image.ndim == 2: image = np.expand_dims(image, axis=0) 
                image_tensor = torch.from_numpy(image).float() / 255.0 
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

            return image_tensor, mask_tensor
    
    def __init__(self, train_image_dir, image_spec, train_image=None, train_model_config=None, training_parameters=None, image_config=None, seg_model_config=None, class_model_config=None, parallel=True):
        """Initializes system structures, data constraints, and dynamic paths using functional configurations.

        Args:
            train_image_dir (str): Base workspace folder directory where processing image directories reside.
            image_spec (str): Dynamic model naming spec string representing target characteristics.
            train_image (dict, optional): Custom execution tracking extensions. Defaults to None.
            train_model_config (dict, optional): Framework specifications configurations dictionary. Defaults to None.
            training_parameters (dict, optional): Parameter dictionary setting layer variables. Defaults to None.
            image_config (dict, optional): Base geometry configuration structures parameter array. Defaults to None.
            seg_model_config (dict, optional): Folder mapping parameters configurations dictionary for segmentation. Defaults to None.
            class_model_config (dict, optional): Folder mapping parameters configurations dictionary for classification. Defaults to None.
            parallel (bool, optional): Standard compute policy toggle. A CUDA GPU is always used when
                available (with multiple GPUs automatically parallelized via `torch.nn.DataParallel`);
                when no GPU is available, CPU-bound work (DataLoader workers) is spread across all
                available CPU cores unless `parallel` is set to False, in which case it runs on a
                single core. Defaults to True.
        """
        _train_image = TRAIN_IMAGE.copy()
        if train_image: _train_image.update(train_image)

        _train_model_config = TRAIN_MODEL_CONFIG.copy()
        if train_model_config: _train_model_config.update(train_model_config)

        _training_parameters = TRAINING_PARAMETERS.copy()
        if training_parameters: _training_parameters.update(training_parameters)

        _image_config = IMAGE_CONFIG.copy()
        if image_config: _image_config.update(image_config)
            
        _seg_model_config = SEG_MODEL_CONFIG.copy()
        if seg_model_config: _seg_model_config.update(seg_model_config)

        _class_model_config = CLASS_MODEL_CONFIG.copy()
        if class_model_config: _class_model_config.update(class_model_config)
        
        self.image_dir = train_image_dir
        self.image_spec = image_spec
        self.mask_dir = os.path.join(self.image_dir, _train_image['mask_subdir'])
        self.split_dir = os.path.join(self.image_dir, _train_image['data_split_subdir'])
        os.makedirs(self.split_dir, exist_ok=True)
        self.split_path = os.path.join(self.split_dir, _train_image['split_filename'])
        
        # Disk-mode classification patch folders, positioned as siblings of the
        # original segmentation training tiles folder (self.image_dir).
        _image_dir_parent = os.path.dirname(os.path.normpath(self.image_dir))
        self.class_manual_track_dir = os.path.join(_image_dir_parent, 'manual_track')
        self.class_manual_bkg_dir = os.path.join(_image_dir_parent, 'manual_bkg')
        
        self.image_extensions = _train_image['image_extensions']
        self.mask_extension = _train_image['mask_extension']
        self.test_split_ratio = _train_image['test_split_ratio']
        self.val_split_ratio = _train_image['val_split_ratio']
        
        self.seg_model_arc = _train_model_config['seg_model_arc']
        self.seg_encoder = _train_model_config['seg_encoder']
        self.seg_encoder_weights = _train_model_config['seg_encoder_weights']
        self.seg_activation = _train_model_config['seg_activation']
        
        self.seg_epochs = _training_parameters['seg_epochs']
        self.seg_min_epochs = _training_parameters['seg_min_epochs']
        self.seg_batch_size = _training_parameters['seg_batch_size']
        self.seg_learning_rate = _training_parameters['seg_learning_rate']
        self.seg_patience = _training_parameters['seg_patience']
        self.seg_threshold = _training_parameters['seg_threshold']
        self.seg_metric_to_monitor = _training_parameters['seg_metric_to_monitor']
        self.seg_val_beta = _training_parameters['seg_val_beta']
        
        self.seg_best_model_folder = _seg_model_config['model_folder_path']
        # Strict Dynamic Naming Scheme Implementation: seg_model_*encoder*_*image_spec*
        self.seg_best_model_spec = f"seg_model_{self.seg_encoder}_{self.image_spec}"
        self.seg_best_model_path = os.path.join(self.seg_best_model_folder, f"{self.seg_best_model_spec}.pth")
        
        self.class_model_type = _train_model_config['class_model_type']
        
        self.class_boost_precision_weight = _training_parameters['class_boost_precision_weight']
        self.class_learning_rate = _training_parameters['class_learning_rate']
        self.class_epochs = _training_parameters['class_epochs']
        self.class_batch_size = _training_parameters['class_batch_size']
        self.class_patience = _training_parameters['class_patience']
        self.class_threshold = _training_parameters['class_threshold']
        
        self.class_best_model_folder = _class_model_config['model_folder_path']
        # Strict Dynamic Naming Scheme Implementation: class_model_*encoder*_*image_spec*
        self.class_best_model_spec = f"class_model_{self.class_model_type}_{self.image_spec}"
        self.class_best_model_path = os.path.join(self.class_best_model_folder, f"{self.class_best_model_spec}.pth")

        os.makedirs(self.seg_best_model_folder, exist_ok=True)
        os.makedirs(self.class_best_model_folder, exist_ok=True)
        
        self.image_height = _image_config['img_height']
        self.image_width = _image_config['img_width']
        self.input_channels_config = _image_config['input_channels_config']
        self.pixel_resolution_um_per_px = _image_config['pixel_resolution_um_per_px']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parallel = parallel
        self.ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        self.seg_efficiency_table = None
        self.seg_cls_efficiency_table = None

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"ERROR: Image directory not found at '{self.image_dir}'")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"ERROR: Mask directory not found at '{self.mask_dir}'")

    def perform_seg_training(self):
        """Executes the standard end-to-end training routine for the semantic segmentation network.

        Returns:
            None
        """
        file_pairs = self._get_training_pairs()
        self._split_dataset(file_pairs)
        self._create_dataloader()
        model = self._create_segmentation_model()
        
        dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=(self.seg_activation is None))
        bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        def loss_fn(pred, target): return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.seg_learning_rate)
        metrics_dict = {
            "FBeta": smp.metrics.fbeta_score, "IoU": smp.metrics.iou_score,
            "Dice/F1": smp.metrics.f1_score, "Recall": smp.metrics.recall, "Precision": smp.metrics.precision,
        }
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
        
        counter = 0
        best_val_metric_value = -1.0
        epoch_logs = []
        
        try:
            for epoch in range(self.seg_epochs):
                print(f"\nEpoch {epoch + 1}/{self.seg_epochs}")
                epoch_train_loss = 0.0

                model.train() 
                train_pbar = tqdm(self.train_loader, desc=f"Train E{epoch+1}", leave=False)
                for images, masks in train_pbar:
                    images, masks = images.to(self.device, dtype=torch.float32), masks.to(self.device, dtype=torch.float32)
                    if images.dim() == 4 and masks.dim() == 3: masks = masks.unsqueeze(1)

                    optimizer.zero_grad()               
                    outputs = model(images)             
                    loss = loss_fn(outputs, masks)      
                    loss.backward()                     
                    optimizer.step()                    

                    epoch_train_loss += loss.item()     
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}")

                avg_train_loss = epoch_train_loss / len(self.train_loader)

                model.eval() 
                epoch_val_loss = 0.0
                epoch_val_scores = {name: 0.0 for name in metrics_dict.keys()}
                val_pbar = tqdm(self.val_loader, desc=f"Val E{epoch+1}", leave=False)

                with torch.no_grad(): 
                    for images, masks in val_pbar:
                        images, masks = images.to(self.device, dtype=torch.float32), masks.to(self.device, dtype=torch.float32)
                        if images.dim() == 4 and masks.dim() == 3: masks = masks.unsqueeze(1)

                        outputs = model(images)
                        loss = loss_fn(outputs, masks)
                        epoch_val_loss += loss.item()

                        for metric_name, metric_fn in metrics_dict.items():
                            predictions = (outputs > self.seg_threshold).int()  
                            tp, fp, fn, tn = (predictions == 1) & (masks == 1), (predictions == 1) & (masks == 0), (predictions == 0) & (masks == 1), (predictions == 0) & (masks == 0)
                            score = metric_fn(tp, fp, fn, tn, beta=self.seg_val_beta, reduction="micro") if metric_name == "FBeta" else metric_fn(tp, fp, fn, tn, reduction="micro")
                            epoch_val_scores[metric_name] += score.item() 

                avg_val_loss = epoch_val_loss / len(self.val_loader)
                for key in epoch_val_scores: epoch_val_scores[key] /= len(self.val_loader)

                epoch_logs.append({
                    "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                    **{f"val_{k.lower()}": v for k, v in epoch_val_scores.items()}
                })

                current_epoch_val_metric = epoch_val_scores.get(self.seg_metric_to_monitor, -1.0) 
                if current_epoch_val_metric > best_val_metric_value:
                    best_val_metric_value = current_epoch_val_metric
                    if self.ngpu > 1:
                        torch.save(model.module.state_dict(), self.seg_best_model_path)  # multi-GPU
                    else:
                        torch.save(model.state_dict(), self.seg_best_model_path)  # single GPU or CPU
                    print(f"Saved dynamic segmentation checkpoint: {self.seg_best_model_path}")
                    counter = 0
                else:
                    counter += 1
                if counter >= self.seg_patience and epoch > self.seg_min_epochs:
                    print("Early stopping triggered")
                    break

                scheduler.step(current_epoch_val_metric)
            
            df = pd.DataFrame(epoch_logs)
            csv_path = os.path.join(self.seg_best_model_folder, f'{self.seg_best_model_spec}_training_metrics.csv')
            df.to_csv(csv_path, index=False)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print("\n--- Training Process Finished ---")
    
    def perform_class_training(self, seg_model_path, seg_th=None, keep_patches_in_memory=False):
        """Extracts candidate structural patches and trains the verification classifier network.

        Args:
            seg_model_path (str): File system source path to the pre-trained segmentation weight dependencies.
            seg_th (float, optional): Custom activation threshold limiting target crop evaluations. Defaults to None.
            keep_patches_in_memory (bool, optional): If True, classification patches are extracted and
                kept as in-memory arrays. If False (default), patches are saved to disk under
                `self.class_manual_track_dir` / `self.class_manual_bkg_dir` and loaded from there.
                Defaults to False.

        Returns:
            None
        """
        file_pairs = self._get_training_pairs()
        self._split_dataset(file_pairs)
        
        train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_ious, val_dices = [], [], [], [], [], [], []
        best_dice = 0

        class_trainset, class_valset = self._create_class_training_dataset(seg_model_path, threshold=seg_th, keep_patches_in_memory=keep_patches_in_memory)
        train_ds = self.TrackDataset(class_trainset, self._get_class_train_augs())
        val_ds = self.TrackDataset(class_valset, get_val_augs())

        train_loader = DataLoader(train_ds, batch_size=self.class_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.class_batch_size, shuffle=False)
        model = self._create_class_model()
        
        BOOST_PRECISION_WEIGHT = torch.tensor([self.class_boost_precision_weight]).to(self.device)  
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=BOOST_PRECISION_WEIGHT)
        optimizer = optim.AdamW(model.parameters(), lr=self.class_learning_rate, weight_decay=1e-3) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.class_patience)

        for epoch in range(self.class_epochs):
            model.train()
            train_loss = 0
            for imgs, labels in tqdm(train_loader, desc=f"Class Train E{epoch+1}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(imgs) 
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            TP, FP, FN, TN = 0, 0, 0, 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > self.class_threshold).float()
                    TP += ((preds == 1) & (labels == 1)).sum().item()
                    FP += ((preds == 1) & (labels == 0)).sum().item()
                    FN += ((preds == 0) & (labels == 1)).sum().item()
                    TN += ((preds == 0) & (labels == 0)).sum().item()

            val_loss /= len(val_loader)
            val_acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            iou = TP / (TP + FP + FN + 1e-8)
            dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_ious.append(iou)
            val_dices.append(dice)

            scheduler.step(val_loss)

            if dice > best_dice:
                best_dice = dice
                if self.ngpu > 1:
                    torch.save(model.module.state_dict(), self.class_best_model_path)  # multi-GPU
                else:
                    torch.save(model.state_dict(), self.class_best_model_path)  # single GPU or CPU
                print(f"Saved dynamic classification checkpoint: {self.class_best_model_path}")
        
        df = pd.DataFrame({
            "train_loss": train_losses, "val_loss": val_losses, "val_accuracy": val_accuracies,
            "val_precision": val_precisions, "val_recall": val_recalls, "val_iou": val_ious, "val_dice": val_dices
        })
        csv_path = os.path.join(self.class_best_model_folder, f"{self.class_best_model_spec}_training_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Classification training completed. Metrics saved at: {csv_path}")
    
    def evaluate_binned_efficiency(self, mode, seg_model_path=None, cls_model_path=None, seg_th=None, cls_th=None, iou_threshold=0.5, num_bins=10, visualize=False):
        """Evaluates detection recalls, precision characteristics, and efficiency distributions.

        Args:
            mode (str): Path execution workflow mode configuration string ('seg' or 'seg_class').
            seg_model_path (str, optional): Alternative path location for segmentation checkpoints. Defaults to None.
            cls_model_path (str, optional): Alternative path location for classification weights. Defaults to None.
            seg_th (float, optional): Custom execution filter ceiling boundaries. Defaults to None.
            cls_th (float, optional): Custom validation layer criteria configurations. Defaults to None.
            iou_threshold (float, optional): Minimum intersection overlap defining structural object hits. Defaults to 0.5.
            num_bins (int, optional): Discretization steps dividing the feature size continuum. Defaults to 10.
            visualize (bool, optional): Activates validation testing plot rendering inside the loops. Defaults to False.

        Returns:
            dict: Calculated execution dictionary logs containing targeted efficiency tables and bins arrays.
            
        Raises:
            ValueError: If an unrecognized mode string identifier parameter is passed.
        """
        if seg_model_path is None: seg_model_path = self.seg_best_model_path
        if cls_model_path is None: cls_model_path = self.class_best_model_path
        if seg_th is None: seg_th = self.seg_threshold
        if cls_th is None: cls_th = self.class_threshold
        
        seg_model_spec = os.path.splitext(os.path.basename(seg_model_path))[0]
        cls_model_spec = os.path.splitext(os.path.basename(cls_model_path))[0]
        
        self.seg_eff_output_spec = "binned_efficiency_" + seg_model_spec + ".csv"
        self.seg_binned_efficiency_path = os.path.join(self.seg_best_model_folder, self.seg_eff_output_spec)
        
        self.seg_cls_eff_output_spec = "binned_efficiency_" + seg_model_spec + "_" + cls_model_spec + ".csv"
        self.seg_cls_binned_efficiency_path = os.path.join(self.class_best_model_folder, self.seg_cls_eff_output_spec)
            
        print(f"\n[OPTIMUS-PRIMUS | EVALUATION PIPELINE] MODE = {mode}")
        with open(self.split_path, "r") as f: splits = json.load(f)
        tot_files = splits["val"] + splits["test"]

        inference_model = self._create_segmentation_model(weights_path=seg_model_path)
        inference_model.eval()
        
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.seg_encoder, self.seg_encoder_weights)
        preprocessing = get_preprocessing(preprocessing_fn, self.image_height, self.image_width)
        class_transform = get_val_augs()

        gt_log, pred_log, pair_log = [], [], []
        val_area_cm2 = 0.
        um_per_px = self.pixel_resolution_um_per_px

        if mode == 'seg':
            for img_path, mask_path in tqdm(tot_files, desc="Evaluating Efficiency"):
                gt_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if um_per_px is not None: val_area_cm2 += gt_mask_full.shape[0] * gt_mask_full.shape[1] * (um_per_px * 1e-4) ** 2
                _, pred_mask_full = self._compute_seg_masks(img_path, inference_model, preprocessing, threshold=seg_th)
                gt_log, pred_log = self._match_instances_by_iou(gt_mask_full, pred_mask_full, iou_threshold, gt_log, pred_log, pair_log)
                
                if visualize:
                    self._plot_verification(cv2.imread(img_path), gt_mask_full, pred_mask_full, f"Seg Evaluation - {os.path.basename(img_path)}")

            efficiency_dict = self._evaluate_efficiency(gt_log, pred_log, num_bins, mode)
            self.seg_efficiency_table = efficiency_dict['efficiency_table']

        elif mode == 'seg_class':
            cls_model = self._create_class_model(cls_weights=cls_model_path)
            for img_path, mask_path in tqdm(tot_files, desc="Evaluating Efficiency"):
                gt_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if um_per_px is not None: val_area_cm2 += gt_mask_full.shape[0] * gt_mask_full.shape[1] * (um_per_px * 1e-4) ** 2
                original_vis_image, pred_mask_full = self._compute_seg_masks(img_path, inference_model, preprocessing, threshold=seg_th)
                final_confirmed_mask = create_class_mask(original_vis_image, pred_mask_full, cls_model, class_transform, self.device, threshold=cls_th)
                gt_log, pred_log = self._match_instances_by_iou(gt_mask_full, final_confirmed_mask, iou_threshold, gt_log, pred_log, pair_log)
                
                if visualize:
                    self._plot_verification(original_vis_image, gt_mask_full, final_confirmed_mask, f"Seg+Class Evaluation - {os.path.basename(img_path)}")
            
            efficiency_dict = self._evaluate_efficiency(gt_log, pred_log, num_bins, mode)
            self.seg_cls_efficiency_table = efficiency_dict['efficiency_table']
        else:
            raise ValueError("mode need to be 'seg' or 'seg_class'")
        
        # Per-instance validation logs, saved next to the binned-efficiency csv, for the detection model
        efficiency_csv = self.seg_binned_efficiency_path if mode == 'seg' else self.seg_cls_binned_efficiency_path
        logs_path = os.path.splitext(efficiency_csv)[0] + "_detection_logs.json"
        self._save_detection_logs(logs_path, gt_log, pred_log, pair_log, val_area_cm2 if um_per_px is not None else None)
        efficiency_dict.update(pair_log=pair_log, val_area_cm2=val_area_cm2, detection_logs_path=logs_path)
        
        return efficiency_dict
    
    def _save_detection_logs(self, path, gt_log, pred_log, pair_log, val_area_cm2):
        """Writes the per-instance validation logs used to calibrate the detection model.

        Args:
            path (str): Output json file.
            gt_log (list): Annotated instances with size and match status.
            pred_log (list): Reported instances with size and match status.
            pair_log (list): (annotated size, reported size) of the matched pairs.
            val_area_cm2 (float or None): Total area of the validation images [cm^2] (None if no pixel calibration).

        Returns:
            None
        """
        payload = {
            'length_unit': 'um' if self.pixel_resolution_um_per_px else 'px',
            'val_area_cm2': None if val_area_cm2 is None else float(val_area_cm2),
            'gt_log': [{'len_um': float(r['len_um']), 'is_true_positive': bool(r['is_true_positive'])} for r in gt_log],
            'pred_log': [{'len_um': float(r['len_um']), 'is_true_positive': bool(r['is_true_positive'])} for r in pred_log],
            'pair_log': [[float(a), float(b)] for a, b in pair_log],
        }
        with open(path, 'w') as f:
            json.dump(payload, f)
        print(f"Detection-model validation logs saved to {path}")
    
    def efficiency_distribution_from_file(self, csv_path=None):
        """Loads calibration distribution datasets directly into internal efficiency execution structures.

        Args:
            csv_path (str, optional): Target file system source path string. Defaults to None.

        Returns:
            None

        Raises:
            FileNotFoundError: If target path is invalid or file missing.
        """
        if not os.path.exists(csv_path): raise FileNotFoundError(f"ERROR: CSV file not found at '{csv_path}'")
        
        if 'segmentation' in csv_path:
            self.seg_efficiency_table = pd.read_csv(csv_path, index_col=0)
        elif 'seg_class' in csv_path:
            self.seg_cls_efficiency_table = pd.read_csv(csv_path, index_col=0)
        
    def _get_training_pairs(self):
        """Builds mapping relationships coordinates between data images and corresponding label masks.

        Returns:
            list: Parsed tuples catalog list matching unique (image_file_path, mask_file_path) pairs.

        Raises:
            FileNotFoundError: If the designated source training directory is empty or missing.
            ValueError: If no valid matching target image-mask pairs can be verified on disk.
        """
        image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        if not image_files: raise FileNotFoundError(f"Error: No image files found in '{self.image_dir}'")

        file_pairs = []
        for img_path in image_files:
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, img_stem + self.mask_extension)
            if os.path.exists(mask_path): file_pairs.append((img_path, mask_path))

        if not file_pairs: raise ValueError("Error: Could not create any image-mask pairs.")
        return file_pairs
    
    def _split_dataset(self, file_pairs):
        """Applies stochastic splits to categorize inputs into unique operational cohorts.

        Args:
            file_pairs (list): Consolidated list array configurations holding image-mask file pairs.

        Returns:
            None
        """
        if os.path.exists(self.split_path):
            with open(self.split_path, "r") as f: splits = json.load(f)
            self.train_files, self.val_files, self.test_files = splits["train"], splits["val"], splits["test"]
        else:
            total_samples = len(file_pairs)
            test_size = int(total_samples * self.test_split_ratio)
            val_size = int(total_samples * self.val_split_ratio)

            if test_size > 0: train_val_files, self.test_files = train_test_split(file_pairs, test_size=test_size, shuffle=True)
            else: train_val_files, self.test_files = file_pairs, []

            if val_size > 0 and len(train_val_files) >= val_size:
                val_split_adjusted = val_size / len(train_val_files)
                self.train_files, self.val_files = train_test_split(train_val_files, test_size=val_split_adjusted, shuffle=True)
            else: self.train_files, self.val_files = train_val_files, []

            splits = {"train": self.train_files, "val": self.val_files, "test": self.test_files}
            with open(self.split_path, "w") as f: json.dump(splits, f, indent=4)
    
    def _create_dataloader(self):
        """Prepares PyTorch streaming data structures supporting active processing execution threads.

        Returns:
            None
        """
        train_augs_pipeline = self._get_seg_train_augs()
        val_test_augs_pipeline = self._get_seg_val_test_augs()
        preprocessing_pipeline = self._get_train_preprocessing()

        train_dataset = self.SegmentationDataset(file_pairs=self.train_files, augmentations=train_augs_pipeline, preprocessing=preprocessing_pipeline, input_channels=self.input_channels_config)
        val_dataset = self.SegmentationDataset(file_pairs=self.val_files, augmentations=val_test_augs_pipeline, preprocessing=preprocessing_pipeline, input_channels=self.input_channels_config)

        num_workers = resolve_num_workers(self.parallel)
        self.train_loader = DataLoader(train_dataset, batch_size=self.seg_batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.seg_batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
        
    def _get_train_preprocessing(self):
        """Resolves target pre-processing transforms depending on segmentation backbone selection.

        Returns:
            A.Compose: Constructed albumentations pre-processing configurations block object.
        """
        try: preprocessing_fn = smp.encoders.get_preprocessing_fn(self.seg_encoder, self.seg_encoder_weights)
        except Exception:
            def basic_preprocessing_fn(image): return image.astype(np.float32) / 255.0
            preprocessing_fn = basic_preprocessing_fn 
        
        _transform = []
        if preprocessing_fn: _transform.append(A.Lambda(image=preprocessing_fn, name="EncoderPreprocessing"))
        _transform.append(ToTensorV2())
        return A.Compose(_transform)
    
    def _get_seg_train_augs(self, augmentation_parameters=AUGMENTATION_PARAMETERS):
        """Constructs stochastic geometrical transformation pipelines for training augmentation.

        Args:
            augmentation_parameters (dict, optional): Baseline probability mappings parameters. Defaults to AUGMENTATION_PARAMETERS.

        Returns:
            A.Compose: Complete Albumentations pipeline composition module framework.
        """
        return A.Compose([
            A.Resize(int(self.image_height), int(self.image_width), interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=augmentation_parameters.get('H_FLIP_PROB', 0.5)),
            A.VerticalFlip(p=augmentation_parameters.get('V_FLIP_PROB', 0.5)),
            A.RandomBrightnessContrast(p=augmentation_parameters.get('BRIGHTNESS_CONTRAST_PROB', 0.4)),
        ])

    def _get_seg_val_test_augs(self):
        """Provides static geometry spatial adjustments matching validation system execution shapes.

        Returns:
            A.Compose: Resizing transform structure targeting base spatial dimensions limits.
        """
        return A.Compose([A.Resize(int(self.image_height), int(self.image_width), interpolation=cv2.INTER_LINEAR)])

    def _create_segmentation_model(self, weights_path=None):
        """Initializes structural segmentation network model architectures inside targeted memory spaces.

        Args:
            weights_path (str, optional): Target checkpoint path file to assign checkpoint values. Defaults to None.

        Returns:
            nn.Module: Built PyTorch segmentation model instance ready for deployment.
        """
        model = smp.create_model(arch=self.seg_model_arc, encoder_name=self.seg_encoder, encoder_weights=self.seg_encoder_weights if weights_path is None else None, in_channels=self.input_channels_config, classes=1, activation=None)
        if weights_path:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"ERROR: Segmentation checkpoint not found at '{weights_path}'")
            model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))

        self.ngpu = torch.cuda.device_count()
        if self.ngpu > 1:
            print(f"Using {self.ngpu} GPUs")
            model = torch.nn.DataParallel(model)

        model.to(self.device)
        return model

    def _create_class_model(self, cls_weights=None):
        """Builds custom patch classification networks targeting specified layer output nodes.

        Args:
            cls_weights (str, optional): Target baseline check-point workspace weight files. Defaults to None.

        Returns:
            nn.Module: Configured classification deep neural network module wrapper.
        """
        model = timm.create_model(self.class_model_type, pretrained=(cls_weights is None), num_classes=1)
        model.conv_stem.stride = (1, 1)
        model.blocks[1][0].conv_dw.stride = (1, 1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
        if cls_weights:
            if not os.path.exists(cls_weights):
                raise FileNotFoundError(f"ERROR: Classification checkpoint not found at '{cls_weights}'")
            state_dict = torch.load(cls_weights, map_location=self.device, weights_only=True)
            if list(state_dict.keys())[0].startswith('module.'): state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        model = model.to(self.device)

        self.ngpu = torch.cuda.device_count()
        if self.ngpu > 1:
            print(f"Using {self.ngpu} GPUs via DataParallel!")
            model = nn.DataParallel(model)

        return model

    def _get_class_train_augs(self):
        """Retrieves stochastic augmentations optimizing target classification data patch structures.

        Returns:
            A.Compose: Configured training transform object mapping sequence variations.
        """
        return A.Compose([
            A.Resize(64, 64), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()
        ])

    def _compute_seg_masks(self, image_path, model, preprocessing, threshold=None):
        """Funnels execution frames through networks to capture target binary response masks.

        Args:
            image_path (str): File image locator string.
            model (nn.Module): Segmentation model tracking workspace logic unit.
            preprocessing (A.Compose): Functional image pixel transforms parameters layer.
            threshold (float, optional): Detection classification confidence ceiling cutoffs. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - image (np.ndarray): Decoded base standard visual RGB frame array matrix.
                - pred_mask (np.ndarray): Extracted binary spatial output evaluation mapping array.
        """
        if threshold is None: threshold = self.seg_threshold
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = preprocessing(image=image)
        preprocessed = sample['image'].unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(preprocessed)
            pred_mask = (torch.sigmoid(pred) > threshold).cpu().numpy().astype(np.uint8).squeeze()
        return image, pred_mask

    def _create_class_training_dataset(self, seg_model_path, threshold=None, keep_patches_in_memory=False):
        """Iterates over split input pools extracting candidate object fragments to compile classification inputs.

        Args:
            seg_model_path (str): File check-point resource mapping directory for segmentation models.
            threshold (float, optional): Detection parsing cutoff constraints coefficients. Defaults to None.
            keep_patches_in_memory (bool, optional): If True, extracted patches are kept as in-memory
                arrays and never written to disk. A warning is printed if the combined train+val patch
                count exceeds `MAX_IN_MEMORY_PATCHES_WARNING`, since this mode holds every patch in RAM
                at once with no eviction. If False (default), patches are saved as PNG files
                to disk under `self.class_manual_track_dir` / `self.class_manual_bkg_dir` (positioned
                as siblings of the original segmentation training tiles folder), split into 'train'
                and 'val' subfolders, and the returned samples reference those file paths instead of
                in-memory arrays. Defaults to False.

        Returns:
            tuple: A tuple of lists containing:
                - train_samples (list): Extracted training samples as (patch, label) tuples, where
                  patch is an in-memory array if `keep_patches_in_memory` is True, or a file path
                  (str) otherwise.
                - val_samples (list): Extracted verification samples in the same format.
        """
        if threshold is None: threshold = self.seg_threshold
        seg_model = self._create_segmentation_model(weights_path=seg_model_path)
        seg_model.eval()
        
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.seg_encoder, self.seg_encoder_weights)
        preprocessing = get_preprocessing(preprocessing_fn, self.image_height, self.image_width)
        
        if not keep_patches_in_memory:
            # Reset the disk-mode classification patch folders before regenerating them.
            for folder in [self.class_manual_track_dir, self.class_manual_bkg_dir]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder)
        
        def extract_patches_from_set(file_list, split_name):
            dataset_samples = []
            
            if not keep_patches_in_memory:
                track_dir = os.path.join(self.class_manual_track_dir, split_name)
                bkg_dir = os.path.join(self.class_manual_bkg_dir, split_name)
                os.makedirs(track_dir, exist_ok=True)
                os.makedirs(bkg_dir, exist_ok=True)
                track_count, bkg_count = 0, 0
            
            for img_path, mask_path in file_list:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, pred_mask = self._compute_seg_masks(img_path, seg_model, preprocessing, threshold=threshold)
                
                for r in regionprops(label(gt_mask > 0)):
                    patch = get_64x64_centered_patch(img, gt_mask, r)
                    if keep_patches_in_memory:
                        dataset_samples.append((patch, 1))
                    else:
                        patch_path = os.path.join(track_dir, f"track_{track_count}.png")
                        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                        dataset_samples.append((patch_path, 1))
                        track_count += 1
                for r in regionprops(label(((pred_mask > 0) & (gt_mask == 0)).astype(np.uint8))):
                    patch = get_64x64_centered_patch(img, pred_mask, r)
                    if keep_patches_in_memory:
                        dataset_samples.append((patch, 0))
                    else:
                        patch_path = os.path.join(bkg_dir, f"bkg_{bkg_count}.png")
                        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                        dataset_samples.append((patch_path, 0))
                        bkg_count += 1
            return dataset_samples

        train_samples = extract_patches_from_set(self.train_files, 'train')
        val_samples = extract_patches_from_set(self.val_files, 'val')

        if keep_patches_in_memory:
            total_patches = len(train_samples) + len(val_samples)
            if total_patches > MAX_IN_MEMORY_PATCHES_WARNING:
                approx_mb = total_patches * 64 * 64 * 3 / (1024 ** 2)
                print(f"Warning: keep_patches_in_memory=True is holding {total_patches} patches "
                      f"(~{approx_mb:.0f} MB) in RAM at once. Consider keep_patches_in_memory=False "
                      f"to stream patches from disk instead if memory becomes a problem.")

        return train_samples, val_samples

    def _match_instances_by_iou(self, gt_mask, pred_mask, iou_threshold, gt_log, pred_log, pair_log=None):
        """
        Match ground truth and predicted instances using a globally-optimal Intersection over Union (IoU)
        assignment (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`).

        Unlike a greedy scan-order match, where an early prediction can lock in a suboptimal ground truth and
        block a later, better-fitting prediction from claiming it, this solves for the assignment that
        maximizes the total IoU over all ground truth / prediction pairs at once, so the result does not
        depend on the (arbitrary) order in which instances are extracted from the masks.

        Args:
        ----------
        gt_mask : np.ndarray
            Ground truth binary mask.

        pred_mask : np.ndarray
            Predicted binary mask.

        iou_threshold : float
            Minimum IoU required to consider a valid match between instances.

        gt_log : list
            List used to accumulate ground truth instance metadata and match status.

        pred_log : list
            List used to accumulate predicted instance metadata and match status.

        pair_log : list, optional
            If given, the (annotated size, reported size) of every matched pair is appended to it;
            used to calibrate the length response of the detection model.

        Returns
        -------
        tuple
            gt_log : list
                Updated ground truth log with match labels.

            pred_log : list
                Updated prediction log with match labels.
        """
        gt_instances = extract_instances(gt_mask)
        pred_instances = extract_instances(pred_mask)

        if gt_instances and pred_instances:
            iou_matrix = np.zeros((len(gt_instances), len(pred_instances)))
            for i, gt in enumerate(gt_instances):
                for j, pred in enumerate(pred_instances):
                    intersection = np.logical_and(gt['mask'], pred['mask']).sum()
                    if intersection == 0:
                        continue
                    union = np.logical_or(gt['mask'], pred['mask']).sum()
                    iou_matrix[i, j] = intersection / union

            # Hungarian algorithm: the assignment that maximizes the total IoU
            # (linear_sum_assignment minimizes the cost, so the IoU matrix is negated).
            gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)

            for i, j in zip(gt_idx, pred_idx):
                if iou_matrix[i, j] >= iou_threshold:
                    gt_instances[i]['matched'] = True
                    pred_instances[j]['matched'] = True
                    if pair_log is not None:
                        pair_log.append((gt_instances[i]['size'], pred_instances[j]['size']))

        for gt in gt_instances:
            gt_log.append({'len_um': gt['size'], 'is_true_positive': gt['matched']})
        for pred in pred_instances:
            pred_log.append({'len_um': pred['size'], 'is_true_positive': pred['matched']})
            
        return gt_log, pred_log

    def _evaluate_efficiency(self, gt_log, pred_log, num_bins, mode):
        """
        Evaluate detection performance by computing size-binned recall and precision of detected tracks.

        Args:
        ----------
        gt_log : list
            List of ground truth instances with size and match information.

        pred_log : list
            List of predicted instances with size and match information.

        num_bins : int
            Number of size bins used for efficiency analysis.

        mode : str
            Inference mode specifying where to store results:
            - 'seg' for segmentation-only results
            - 'seg_class' for segmentation + classification results

        Returns
        -------
        pandas.DataFrame
            Table containing binned recall, precision, counts, and bin centers.

        Raises
        ------
        ValueError
            If `mode` is not 'seg' or 'seg_class'.
        """
        df_gt = pd.DataFrame(gt_log)
        df_pred = pd.DataFrame(pred_log)

        if df_gt.empty or df_pred.empty:
            print("Warning: No instances found in Ground Truth or Predictions.")
            return None
        
        precision = df_pred['is_true_positive'].mean()
        recall = df_gt['is_true_positive'].mean()
        
        print(f"Overall Recall :{df_gt['is_true_positive'].mean()}")
        print(f"Overall Precision : {df_pred['is_true_positive'].mean()}")
        print(f"True Positive : {df_gt['is_true_positive'].sum()}")
        print(f"False Positive : {len(df_pred['is_true_positive']) - df_pred['is_true_positive'].sum()}")
        print(f"False Negative : {len(df_gt['is_true_positive']) - df_gt['is_true_positive'].sum()}")        
        
        df_gt['size_bin'], bins = pd.qcut(df_gt['len_um'], q=num_bins, retbins=True, duplicates='drop')

        df_pred['size_bin'] = pd.cut(df_pred['len_um'], bins=bins, include_lowest=True)
        
        recall_df = df_gt.groupby('size_bin', observed=False)['is_true_positive'].agg( Recall='mean', GT_Count='count')
        precision_df = df_pred.groupby('size_bin', observed=False)['is_true_positive'].agg( Precision='mean',  Pred_Count='count' )
        
        bins_mid = (bins[:-1] + bins[1:]) / 2
        
        efficiency_table = pd.concat([recall_df, precision_df], axis=1)
        efficiency_table.index.name = f'Size Bin ({ "um" if self.pixel_resolution_um_per_px else "px" })'
        
        efficiency_table['Bin_mid'] = bins_mid
        
        efficiency_table = efficiency_table.fillna(0)
        
        if mode == 'seg' :
            efficiency_table.to_csv(self.seg_binned_efficiency_path)
            self.seg_efficiency_table = efficiency_table
            print(f"Overall segmentation Recall :{recall}")
            print(f"Overall segmentation Precision : {precision}")
        elif mode == 'seg_class' :
            efficiency_table.to_csv(self.seg_cls_binned_efficiency_path)
            self.seg_cls_efficiency_table = efficiency_table
            print(f"Overall segmentation + classification Recall :{recall}")
            print(f"Overall segmentation + classification Precision : {precision}")
        else :
            raise ValueError("mode need to be 'seg' or 'seg_class'")
        
        return {'precision': precision,
                'recall': recall,
                'efficiency_table': efficiency_table,
                'df_gt': df_gt,
                'df_pred': df_pred,
                'bins': bins
            }

    def _plot_verification(self, img, gt, pred, title_text):
        """Displays interactive execution verification grids using active visual frame overlays.

        Args:
            img (np.ndarray): Source matrix standard background configuration RGB array frame.
            gt (np.ndarray): Binary truth channel layout overlay dimensions matrix.
            pred (np.ndarray): System extracted tracking layer evaluation array prediction layout matrix.
            title_text (str): Identification context tag string tracking image layout headers.

        Returns:
            None
        """
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Canvas")
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(gt, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(title_text)
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        plt.show()