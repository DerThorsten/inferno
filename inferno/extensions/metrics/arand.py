from .base import Metric
import numpy as np
import scipy.sparse as sparse
import logging


class ArandScore(Metric):
    """Arand Score, as defined in [1].

    References
    ----------
    [1]: http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full#h3
    """
    def __init__(self, average_slices=True):
        self.average_slices = average_slices

    # compute the arand score for a prediction target pair
    def _arand_for_tensor(self, prediction, target):
        ndim = prediction.ndim
        average_slices = self.average_slices and ndim == 3

        if average_slices:
            # average the arand values over the 3d slices
            evaluation_values = [adapted_rand(pred, targ)
                                 for pred, targ in zip(prediction, target)]
            return np.mean([eval_val[0] for eval_val in evaluation_values if eval_val is not None])
        else:
            return adapted_rand(prediction, target)[0]

    def forward(self, prediction, target):
        assert(prediction.shape == target.shape), "%s, %s" % (str(prediction.shape),
                                                              str(target.shape))
        assert prediction.shape[1] == 1, "Expect singleton channel axis"
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()

        ndim = prediction.ndim
        assert ndim in (4, 5), "Expect 2 or 3d input with additional batch and channel axis"

        # return the average arand error over the batches
        return np.mean([self._arand_for_tensor(pred[0], targ[0])
                        for pred, targ in zip(prediction, target)])


class ArandError(ArandScore):
    """Arand Error = 1 - <arand score>"""
    def __init__(self, **super_kwargs):
        super(ArandError, self).__init__(**super_kwargs)

    def forward(self, prediction, target):
        return 1. - super(ArandError, self).forward(prediction, target)


# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py
def adapted_rand(seg, gt):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.

    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label

    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision.
    rec : float, optional
        The adapted Rand recall.

    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    logger = logging.getLogger(__name__)

    assert seg.shape == gt.shape, "%s, %s" % (str(seg.shape), str(gt.shape))

    if np.any(seg == 0):
        logger.debug("Zeros in segmentation, treating as background.")
    if np.any(gt == 0):
        logger.debug("Zeros in ground truth, 0's will be ignored.")

    seg_zeros = np.all(seg == 0)
    gt_zeros = np.all(gt == 0)
    if  seg_zeros or gt_zeros:
        if seg_zeros:
            logger.warning("Segmentation is all zeros, ignoring for eval.")
            return None
        else:
            print(gt.shape)
            print(np.unique(gt))
            logger.warning("Groundtruth is all zeros, ignoring for eval.")
            return None

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)

    # mask to foreground in A
    mask = (segA > 0)
    segA = segA[mask]
    segB = segB[mask]

    # number of nonzero pixels in original segA
    n = segA.size
    n_labels_A = int(np.amax(segA)) + 1
    n_labels_B = int(np.amax(segB)) + 1

    ones_data = np.ones(n)
    p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                             shape=(n_labels_A, n_labels_B),
                             dtype=np.uint64)

    # In the paper where adapted rand is proposed, they treat each background
    # pixel in segB as a different value (i.e., unique label for each pixel).
    # To do this, we sum them differently than others

    # ind (label_gt, label_seg), so ignore 0 seg labels
    B_nonzero = p_ij[:, 1:]
    B_zero = p_ij[:, 0]

    # this is a count
    num_B_zero = B_zero.sum()

    # sum of the joint distribution
    #   separate sum of B>0 and B=0 parts
    sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero

    # these are marginal probabilities
    # sum over all seg labels overlapping one gt label (except 0 labels)
    a_i = p_ij.sum(1)
    b_i = B_nonzero.sum(0)

    sum_a = np.power(a_i, 2).sum()
    sum_b = np.power(b_i, 2).sum() + num_B_zero

    precision = float(sum_p_ij) / sum_b
    recall = float(sum_p_ij) / sum_a
    f_score = 2.0 * precision * recall / (precision + recall)
    return f_score, precision, recall
