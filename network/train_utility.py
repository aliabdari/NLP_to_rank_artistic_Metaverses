import os
import pickle
import numpy as np
import torch


def get_entire_data():
    data_description_ = '../open_clip_features/descriptions/sentences'
    data_scene_ = torch.load('../scene_features/final_tensor_scenes_bedroom.pt')
    data_img_ = '../open_clip_features/images'
    return data_description_, data_scene_, data_img_


def create_rank(result, entire_descriptor, desired_output_index):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    position = torch.where(sorted_indices == desired_output_index)
    return position[0].item(), sorted_indices


def evaluate(output_description, output_scene, section):
    avg_rank_scene = 0
    ranks_scene = []
    avg_rank_description = 0
    ranks_description = []

    ndcg_10_list = []
    ndcg_entire_list = []

    for j, i in enumerate(output_scene):
        rank, sorted_list = create_rank(i, output_description, j)
        avg_rank_scene += rank
        ranks_scene.append(rank)

    for j, i in enumerate(output_description):
        rank, sorted_list = create_rank(i, output_scene, j)
        avg_rank_description += rank
        ranks_description.append(rank)

    ranks_scene = np.array(ranks_scene)
    ranks_description = np.array(ranks_description)

    n_q = len(output_scene)
    sd_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
    sd_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
    sd_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
    sd_r50 = 100 * len(np.where(ranks_scene < 50)[0]) / n_q
    sd_r100 = 100 * len(np.where(ranks_scene < 100)[0]) / n_q
    sd_medr = np.median(ranks_scene) + 1
    sd_meanr = ranks_scene.mean() + 1

    n_q = len(output_description)
    ds_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
    ds_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
    ds_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
    ds_r50 = 100 * len(np.where(ranks_description < 50)[0]) / n_q
    ds_r100 = 100 * len(np.where(ranks_description < 100)[0]) / n_q
    ds_medr = np.median(ranks_description) + 1
    ds_meanr = ranks_description.mean() + 1

    ds_out, sc_out = "", ""
    for mn, mv in [["R@1", ds_r1],
                   ["R@5", ds_r5],
                   ["R@10", ds_r10],
                   ["R@50", ds_r50],
                   ["R@100", ds_r100],
                   ["median rank", ds_medr],
                   ["mean rank", ds_meanr],
                   ]:
        ds_out += f"{mn}: {mv:.4f}   "

    for mn, mv in [("R@1", sd_r1),
                   ("R@5", sd_r5),
                   ("R@10", sd_r10),
                   ("R@50", sd_r50),
                   ("R@100", sd_r100),
                   ("median rank", sd_medr),
                   ("mean rank", sd_meanr),
                   ]:
        sc_out += f"{mn}: {mv:.4f}   "

    print(section + " data: ")
    print("Scenes ranking: " + ds_out)
    print("Descriptions ranking: " + sc_out)
    if section == "test" and len(ndcg_10_list) > 0:
        avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
        avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
    else:
        avg_ndcg_10_entire = -1
        avg_ndcg_entire = -1

    return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


def retrieve_indices(data_size):
    if os.path.isfile('./indices/indices.pkl'):
        indices_pickle = open('indices/indices.pkl', "rb")
        indices_pickle = pickle.load(indices_pickle)
        train_indices = indices_pickle["train"]
        val_indices = indices_pickle["val"]
        test_indices = indices_pickle["test"]
    else:
        train_ratio = .7
        val_ratio = .15
        perm = torch.randperm(data_size)

        train_indices_tmp = perm[:int(data_size * train_ratio)]
        train_indices = [num * 10 + i for num in train_indices_tmp for i in range(10)]
        train_indices = torch.tensor(train_indices)

        val_indices_tmp = perm[int(data_size * train_ratio):int(data_size * (val_ratio + train_ratio))]
        val_indices = [num * 10 + i for num in val_indices_tmp for i in range(10)]
        val_indices = torch.tensor(val_indices)

        test_indices_tmp = perm[int(data_size * (val_ratio + train_ratio)):]
        test_indices = [num * 10 + i for num in test_indices_tmp for i in range(10)]
        test_indices = torch.tensor(test_indices)

        indices_pickle = {"train": train_indices, "val": val_indices, "test": test_indices}
        with open('./indices/indices.pkl', 'wb') as f:
            pickle.dump(indices_pickle, f)

    return train_indices, val_indices, test_indices


def save_best_model(best_model_state_dict_scene, best_model_state_dict_description, model_name,
                    best_model_state_dict_converter=None):
    model_path = "models"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = model_path + os.sep + model_name
    if best_model_state_dict_converter is None:
        torch.save({'model_state_dict_scene': best_model_state_dict_scene,
                    'model_state_dict_description': best_model_state_dict_description},
                   model_path)
    else:
        torch.save({'model_state_dict_scene': best_model_state_dict_scene,
                    'model_state_dict_description': best_model_state_dict_description,
                    'model_state_dict_converter': best_model_state_dict_converter},
                   model_path)


def load_best_model(model_name):
    model_path = "models"
    model_path = model_path + os.sep + model_name
    check_point = torch.load(model_path)
    best_model_state_dict_scene = check_point['model_state_dict_scene']
    best_model_state_dict_description = check_point['model_state_dict_description']
    if check_point.get('model_state_dict_converter') is None:
        return best_model_state_dict_scene, best_model_state_dict_description
    else:
        best_model_state_dict_converter = check_point['model_state_dict_converter']
        return best_model_state_dict_scene, best_model_state_dict_description, best_model_state_dict_converter


def cosine_sim(im, s):
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim
