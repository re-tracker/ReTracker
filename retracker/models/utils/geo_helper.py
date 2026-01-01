import torch


def _rotate(image):
    image = image.permute(0, 1, 3, 2)
    rotated = torch.flip(image, dims=[2])
    return rotated


def _rotate_coords(coords, W):
    x, y = coords[..., :1], coords[..., 1:2]
    x_new = y
    y_new = W - 1 - x
    coords = torch.cat([x_new, y_new], dim=-1)
    return coords


def _rotate_coords_back(coords, W):
    x, y = coords[..., :1], coords[..., 1:2]
    y_new = x
    x_new = W - 1 - y
    coords = torch.cat([x_new, y_new], dim=-1)
    return coords
