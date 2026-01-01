import torch

from ..utils.misc import normalize_keypoints


# preprocess different types of input
def _preprocess_data(data):
    """prepare image and queries"""

    data.update(
        {
            "bs": data["images"].shape[0] if "images" in data else data["image0"].shape[0],
            "hw_i": data["images"].shape[3:] if "images" in data else data["image0"].shape[2:],
        }
    )
    if "images" not in data.keys():
        data.update(
            {
                "images": torch.stack([data["image0"], data["image1"]], dim=1),
            }
        )

    if "queries" not in data:
        if "trajs" in data:
            data["queries"] = data["trajs"][:, 0]
        else:
            raise NotImplementedError
    queries = data["queries"]
    queries_num = queries.shape[1]
    data.update({"queries_real_num": queries_num})

    queries_norm_pos = normalize_keypoints(data["queries"], H=data["hw_i"][0], W=data["hw_i"][1])
    data.update({"queries_norm_pos": queries_norm_pos})
