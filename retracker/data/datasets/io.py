import io

import cv2
import numpy as np
import h5py
import torch
from kornia.geometry import homography_warp, normal_transform_pixel, normalize_homography
from PIL import Image
from numpy.linalg import inv

from retracker.utils.rich_utils import CONSOLE
# NOTE: This file is open-source friendly and only supports local filesystem paths.
# If you have datasets stored in remote/object storage, download them locally (or
# add your own adapter in a private fork).

# --- DATA IO ---
def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        if isinstance(resize, int):
            scale = resize / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        else: # omegaconf.listconfig.ListConfig
            w_new, h_new = resize
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new

def imread_gray(path, augment_fn=None):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith("oss://"):
        raise ValueError(f"Unsupported path scheme (oss://). Please download locally: {path}")

    image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def imread_color(path, augment_fn=None):
    if str(path).startswith("oss://"):
        raise ValueError(f"Unsupported path scheme (oss://). Please download locally: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if augment_fn is not None:
        image = augment_fn(image)
    return image  # (h, w)


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
        mask = mask[0]
    else:
        raise NotImplementedError()
    return padded, mask


def _corners_xy(w: int, h: int) -> np.ndarray:
    """Return image corners in (x, y) order, shape [4, 2]."""
    return np.array(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )


def sample_homography_sap(h: int, w: int, max_jitter: float = 0.1) -> np.ndarray:
    """Sample a simple random homography by jittering the four image corners.

    This is a lightweight, dependency-free replacement for internal augmentation
    helpers that were present in earlier iterations of the project.

    Returns:
        3x3 homography (float64) mapping src -> dst.
    """
    src = _corners_xy(w, h)
    jitter = float(max_jitter) * float(min(h, w))
    dst = src + np.random.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
    dst[:, 0] = np.clip(dst[:, 0], 0.0, float(w - 1))
    dst[:, 1] = np.clip(dst[:, 1], 0.0, float(h - 1))
    return cv2.getPerspectiveTransform(src, dst)


def sample_homography_corners(
    src_size_wh: list[int] | tuple[int, int],
    dst_size_wh: list[int] | tuple[int, int],
    max_jitter: float = 0.1,
):
    """Sample a homography and return the src/dst corner coordinates.

    Return signature matches historical call sites:
        (H, src_corners, dst_corners, meta)
    """
    src_w, src_h = int(src_size_wh[0]), int(src_size_wh[1])
    dst_w, dst_h = int(dst_size_wh[0]), int(dst_size_wh[1])
    if (src_w, src_h) != (dst_w, dst_h):
        raise ValueError("sample_homography_corners currently supports same src/dst sizes only")

    src = _corners_xy(src_w, src_h)
    jitter = float(max_jitter) * float(min(src_h, src_w))
    dst = src + np.random.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
    dst[:, 0] = np.clip(dst[:, 0], 0.0, float(src_w - 1))
    dst[:, 1] = np.clip(dst[:, 1], 0.0, float(src_h - 1))
    H = cv2.getPerspectiveTransform(src, dst)
    return H, src, dst, None


# --- MEGADEPTH ---
def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    if read_gray:
        image = imread_gray(path, augment_fn)
    else:
        image = imread_color(path, augment_fn)

    # resize image
    try:
        w, h = image.shape[1], image.shape[0]
    except Exception:
        CONSOLE.print(f"[red]Failed to read image: {path}[/red]")
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    if not read_gray:
        image = image.transpose(2,0,1)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    return image, mask, scale, origin_img_size


def read_megadepth_gray_sample_homowarp(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    if read_gray:
        image = imread_gray(path, augment_fn)
    else:
        image = imread_color(path, augment_fn)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)
    
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    # Sample homography and warp:
    homo_sampled = sample_homography_sap(h, w) # 3*3
    homo_sampled_normed = normalize_homography(
        torch.from_numpy(homo_sampled[None]).to(torch.float32),
        (h, w),
        (h, w),
    )

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None, None] / 255 # B * C * H * W
    else:
        image = torch.from_numpy(image).float().permute(2,0,1)[None] / 255

    homo_warpped_image = homography_warp(
        image, # 1 * C * H * W
        torch.linalg.inv(homo_sampled_normed),
        (h, w),
    )
    image = (homo_warpped_image[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    norm_pixel_mat = normal_transform_pixel(h, w) # 1 * 3 * 3

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if not read_gray:
        image = image.transpose(2,0,1)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    return image, mask, scale, origin_img_size, norm_pixel_mat[0], homo_sampled_normed[0]


def read_megadepth_gray_sample_homowarp_by_glue(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    if read_gray:
        image = imread_gray(path, augment_fn)
    else:
        image = imread_color(path, augment_fn)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)
    
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    # Sample homography and warp:
    H, _, coords, _ = sample_homography_corners([w, h], [w, h])
    image = cv2.warpPerspective(image, H, (w, h))

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if not read_gray:
        image = image.transpose(2,0,1)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    return image, mask, scale, origin_img_size, H, coords


def read_megadepth_depth_gray(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    if '.png' in path:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])

    # following controlnet  1-depth
    depth = depth.astype(np.float64)
    depth_non_zero = depth[depth!=0]
    vmin = np.percentile(depth_non_zero, 2)
    vmax = np.percentile(depth_non_zero, 85)
    depth -= vmin
    depth /= (vmax - vmin + 1e-4)
    depth = 1.0 - depth
    image = (depth * 255.0).clip(0, 255).astype(np.uint8)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if read_gray:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = np.stack([image]*3) # 3 * H * W
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    return image, mask, scale, origin_img_size


def read_megadepth_depth(path, pad_to=None):
    if str(path).startswith("oss://"):
        raise ValueError(
            f"Unsupported path scheme (oss://). Please download locally and pass a filesystem path: {path}"
        )

    if "h5" in str(path):
        depth = np.array(h5py.File(path, "r")["depth"])
    elif "png" in str(path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        depth = depth / 1000

    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_depth(path):
    if str(path).startswith("s3://"):
        raise ValueError(
            f"Unsupported path scheme (s3://). Please download locally and pass a filesystem path: {path}"
        )

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]
