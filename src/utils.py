import numpy as np
import ruptures as rpt

UNLABELLED_IDX = -1


def create_labels(bkps_list, annotation_ratio=1.0):
    """Create annotations centered inside each breaking points. Non annotated part have a -1 label

    Args:
        bkps_list (list): list of indexes of breaking points
        annotation_ratio (float, optional): ratio of labelled part. Defaults to 1.0.

    Returns:
        list: labels
    """
    labels_list = []
    for bkps in bkps_list:
        bkps = [0] + sorted(bkps)
        labels = np.full((bkps[-1], 1), UNLABELLED_IDX)
        for idx, (start, end) in enumerate(rpt.utils.pairwise(bkps)):
            offset = int((end - start) * (1 - annotation_ratio) // 2)
            labels[start + offset : end - offset] = idx
        labels_list.append(labels)
    return labels_list


def compute_f1(precision, recall):
    return 2 * (recall * precision) / (recall + precision)

def evaluate_cross_val(cost, signals, bkps_list, prefix, search_method,pen,lambd):
    hausdorff_score = []
    f1_score = []
    moy= []
    pelt_lambda=Pelt_lambda(custom_cost=cost)
    pelt_lambda.calcul_penality(signals,bkps_list)
    for i in range(len(signals)):

        pelt_lambda.fit(signals[i])
        d=signals[i][0].shape[0]
        print(f"d = {d}")
        bkps_predicted = pelt_lambda.predict()
        print(pen(lambd,calcul_features(signals[i]),d))
        print(bkps_predicted)
        print(bkps_list[i])
        moy.append((len(bkps_predicted)-1)/(len(bkps_list[i])-1))
        if len(bkps_predicted)!=1:
            hausdorff_score.append(hausdorff(bkps_predicted, bkps_list[i]))
        precision, recall = precision_recall(bkps_list[i], bkps_predicted)
        if precision+recall != 0:
            f1_score.append(compute_f1(precision, recall))

    hausdorff_score = np.mean(hausdorff_score)
    f1_score = np.mean(f1_score)
    fig, ax_array = rpt.display(signals[-1], bkps_list[-1], bkps_predicted)
    print(f"{prefix}_hausdorff: {hausdorff_score:.3f}" f" {prefix}_f1: {f1_score:.3f}" f" Moyenne :{np.mean(moy)}")
    plt.show()

    return hausdorff_score,f1_score,np.mean(moy)