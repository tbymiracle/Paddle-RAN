import numpy as np
from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./forward_paddle.npy")
    info2 = diff_helper.load_info("./forward_torch.npy")
    info3 = diff_helper.load_info("./loss_paddle.npy")
    info4 = diff_helper.load_info("./loss_torch.npy")
    info5 = diff_helper.load_info("./backward_paddle.npy")
    info6 = diff_helper.load_info("./backward_torch.npy")

    diff_helper.compare_info(info1, info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-5, path="./forward_diff.txt")

    diff_helper.compare_info(info3, info4)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-5, path="./loss_diff.txt")

    diff_helper.compare_info(info5, info6)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./backward_diff.txt")