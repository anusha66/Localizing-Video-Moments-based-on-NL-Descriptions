import pickle as p
import numpy as np
import pdb

rgb_file = open('../results/single_gif/clip_retrieval_rgb_baseline_single_gif_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2_test.p', 'rb')
rgb_scores = p.load(rgb_file)

flow_file = open('../results/single_gif/clip_retrieval_flow_baseline_single_gif_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2_test.p', 'rb')
flow_scores = p.load(flow_file)

activity_file = open('activity_recog1.pickle', 'rb')
activity_scores = p.load(activity_file)

npa = np.array
def get_soft_max_scores(w, t=1.0):
    e = np.exp(npa(w) / t)
    softmax = e/np.sum(e)
    return softmax


def iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


def rank(pred, gt):
    return pred.index(tuple(gt)) + 1


possible_segments = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]



average_ranks = []
average_iou = []
for score in activity_scores:
    annot_id = score[0]
    activity_score_list = score[1]
    gt = score[2]

    act_sm = get_soft_max_scores(activity_score_list)[:-1]
    rgb_sm = get_soft_max_scores(rgb_scores[30000][annot_id])
    flo_sm = get_soft_max_scores(flow_scores[30000][annot_id])


    # act_sm = activity_score_list[:-1]
    # rgb_sm = rgb_scores[30000][annot_id]
    # flo_sm = flow_scores[30000][annot_id]

    # pdb.set_trace()
    scores = (0.2)*act_sm + (0.4)*rgb_sm + (0.4)*flo_sm

    # scores = rgb_sm + flo_sm

    # pdb.set_trace()

    seg_idx = [np.argsort(scores)][0]

    segments = []
    for idx in seg_idx:
    	segments.append(possible_segments[idx])
    pred = segments[0]
    ious = [iou(pred, t) for t in gt]
    average_iou.append(np.mean(np.sort(ious)[-3:]))
    ranks = [rank(segments, t) for t in gt]
    average_ranks.append(np.mean(np.sort(ranks)[:3]))
    
pdb.set_trace()
rank1 = np.sum(np.array(average_ranks) <= 1)/float(len(average_ranks))
rank5 = np.sum(np.array(average_ranks) <= 5)/float(len(average_ranks))
miou = np.mean(average_iou)


print ('rank@1', rank1)
print ('rank@5', rank5)
print ('miou', miou)






