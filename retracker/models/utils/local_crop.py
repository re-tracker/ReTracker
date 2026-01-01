import torch
from einops import rearrange

from retracker.models.utils.misc import bilinear_sampler


class LocalCrop:
    def __init__(self, corr_num_levels=3, corr_radius=3):
        self.corr_num_levels = corr_num_levels
        self.corr_radius = corr_radius

    def get_support_points(self, coords, r, reshape_back=True):
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)

        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)

        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl

        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def get_correlation_feat(self, fmaps, queried_coords, r=3):
        """
        fmaps: B F C H W
        queried_coords: BF N C=2
        """
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(fmaps.reshape(B * T, D, 1, H_, W_), support_points)
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )  # B T N W W C

    def get_track_feat_3lvl(self, fmaps_pyramid, queries, scales=None, fmaps_scale=2):
        """crop features from fmap_pyramid
        Args:
            fmaps: B F C H W
            queries: BF N C=2
        Returns:
            corr_feat_nlvl: list [BN F C W W ]
        """
        if scales is None:
            scales = (16, 8, 2)
        corr_feat_nlvl = []
        queries = queries / 2.0  # init queries level is 2x
        coords_init = queries

        for i in range(self.corr_num_levels):
            radius_list = [self.corr_radius] * self.corr_num_levels
            corr_feat_i = self.get_correlation_feat(
                fmaps_pyramid[i], coords_init * 2 / scales[i], r=radius_list[i]
            )
            corr_feat_i = rearrange(corr_feat_i, "B T N W V C -> (B N) T C W V")
            corr_feat_nlvl.append(corr_feat_i)

        return corr_feat_nlvl

    def get_track_feat_single_level(self, fmap, queries, scale, fmaps_scale=2):
        """Crop features from a single level of feature map.

        This is more efficient than get_track_feat_3lvl when you only need
        features from one level at a time.

        Args:
            fmap: B F C H W (single level feature map)
            queries: BF N C=2 (query coordinates at 2x scale)
            scale: int, the scale of this level (16, 8, or 2)

        Returns:
            corr_feat: [BN F C W V] features at the specified level
        """
        queries = queries / 2.0  # init queries level is 2x
        coords_init = queries

        corr_feat = self.get_correlation_feat(fmap, coords_init * 2 / scale, r=self.corr_radius)
        corr_feat = rearrange(corr_feat, "B T N W V C -> (B N) T C W V")
        return corr_feat

    def get_crop_patches(self, fmaps, queries, scales=None, fmaps_scale=1):
        """crop features from fmap_pyramid
        Args:
            fmaps: B C H W
            queries: B N C=2
        Returns:
            corr_feat_nlvl: list [B N C W W ]
        """
        if scales is None:
            scales = (16, 8, 2)
        corr_feat_nlvl = []
        centroids = queries

        fmaps_pyramid = [
            torch.nn.functional.interpolate(fmaps, scale_factor=1.0 / scale_i, mode="nearest")[
                :, None
            ]
            for scale_i in scales
        ]

        for i in range(len(scales)):
            patches_i = self.get_correlation_feat(
                fmaps_pyramid[i], centroids / scales[i], r=self.corr_radius
            )
            patches_i = rearrange(patches_i, "B T N W V C -> (B T) N C W V")  # T=1
            corr_feat_nlvl.append(patches_i)

        return corr_feat_nlvl
