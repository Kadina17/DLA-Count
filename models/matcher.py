import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher_Crowd(nn.Module):
    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.k_nearest = 5  # 最近邻点的数量
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["point"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L2 cost between point
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        # 计算所有预测点与所有真值点的距离矩阵
        distances = torch.cdist(out_points, tgt_points, p=2)  # [batch_size * num_queries, num_target_points]

        # 找到每个预测点的k个最近邻点
        k = min(self.k_nearest, distances.size(1))  # 确保k不超过真值点数量
        nearest_distances, _ = torch.topk(distances, k, dim=1, largest=False)  # [batch_size * num_queries, k]

        # 计算每个预测点的局部邻域范围（动态delta）
        dynamic_deltas = nearest_distances.mean(dim=1, keepdim=True)  # [batch_size * num_queries, 1]

        # 找到局部邻域内的点（距离小于dynamic_delta）
        in_neighborhood = distances < dynamic_deltas  # [batch_size * num_queries, num_target_points]

        # 计算高斯权重（向量化操作）
        weights = torch.exp(-distances ** 2 / (2 * 5.0 ** 2))  # sigma=5.0
        cost_point = torch.where(in_neighborhood, distances * weights, distances)

        # Final cost matrix
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["point"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def gaussian_weight(distance, sigma):
    """
    计算高斯衰减权重
    :param distance: 预测点与真值点之间的欧氏距离
    :param sigma: 高斯衰减系数
    :return: 高斯权重值
    """
    return torch.exp(-distance**2 / (2 * sigma**2))


def build_matcher_crowd(args):
    return HungarianMatcher_Crowd(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
