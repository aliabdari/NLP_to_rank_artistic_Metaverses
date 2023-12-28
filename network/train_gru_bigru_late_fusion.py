from tqdm import tqdm
from DNNs import FCNet, GRUNet, ConverterNN
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from Data_utils import DescriptionSceneDatasetV2
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import train_utility


def contrastive_loss(pairwise_distances, targets, margin=0.25):
    batch_size = pairwise_distances.shape[0]
    diag = pairwise_distances.diag().view(batch_size, 1)
    pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
    d1 = diag.expand_as(pairwise_distances)
    cost_s = (margin + pairwise_distances - d1).clamp(min=0)
    cost_s = cost_s.masked_fill(pos_masks, 0)
    cost_s = cost_s / (batch_size * (batch_size - 1))
    cost_s = cost_s.sum()

    d2 = diag.t().expand_as(pairwise_distances)
    cost_d = (margin + pairwise_distances - d2).clamp(min=0)
    cost_d = cost_d.masked_fill(pos_masks, 0)
    cost_d = cost_d / (batch_size * (batch_size - 1))
    cost_d = cost_d.sum()

    return (cost_s + cost_d) / 2


def collate_fn(data):
    # desc
    tmp_data = [x[0] for x in data]
    tmp = pad_sequence(tmp_data, batch_first=True)
    desc_ = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tmp_data]),
                                 batch_first=True,
                                 enforce_sorted=False)
    # scenes
    tmp_scene = [x[1] for x in data]
    scenes_ = torch.stack(tmp_scene)

    # imgs
    tmp_imgs = [x[2] for x in data]
    imgs_ = torch.stack(tmp_imgs)

    return desc_, scenes_, imgs_


def start_train():
    type_room = 'bedroom'
    print("type_room:", type_room)

    output_feature_size = 256

    is_bidirectional = True
    model_descriptor = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_scene = FCNet(input_size=200, feature_size=128)
    model_converter = ConverterNN(dim=512, out_dim=128)

    num_epochs = 30
    batch_size = 64

    # Loading Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_descriptor.to(device)
    model_scene.to(device)
    model_converter.to(device)

    # Loading Data
    # if type_room in ['living', 'bedroom']:
    data_description_, data_scene_, data_img_ = train_utility.get_entire_data()
    train_indices, val_indices, test_indices = train_utility.retrieve_indices(3384)
    dataset = DescriptionSceneDatasetV2(data_description_, data_scene_, data_img_, type_model_desc="gru")

    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    params = list(model_descriptor.parameters()) + list(model_scene.parameters()) + list(model_converter.parameters())
    optimizer = torch.optim.Adam(params, lr=0.008)

    train_losses = []
    val_losses = []

    # Define the StepLR scheduler
    step_size = 17
    gamma = 0.75
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    if is_bidirectional:
        file_prefix = "bidirectional_gru_" + type_room
    else:
        file_prefix = "gru_" + type_room

    r10_hist = []
    best_r10 = 0

    for epoch in tqdm(range(num_epochs)):
        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size).to(device)
        output_scene_val = torch.empty(len(val_indices), output_feature_size).to(device)

        for i, (data_description, data_scene, data_img) in enumerate(train_loader):
            data_scene = data_scene.to(device)
            data_description = data_description.to(device)
            data_img = data_img.to(device)

            # zero the gradients
            optimizer.zero_grad()

            output_descriptor = model_descriptor(data_description)
            output_converter = model_converter(data_img)
            output_scene_ = model_scene(data_scene)
            output_scene = torch.cat((output_scene_, output_converter), dim=1)

            # transposed_scene_features = torch.transpose(output_scene, 0, 1)
            # multiplication = torch.mm(output_descriptor, transposed_scene_features)
            multiplication = train_utility.cosine_sim(output_descriptor, output_scene)

            # print(multiplication.diagonal().mean().item())
            # define the target
            ground_truth = torch.eye(multiplication.size()[0], device=device)

            loss = contrastive_loss(multiplication, targets=ground_truth)

            loss.backward()

            optimizer.step()

            total_loss_train += loss.item()
            num_batches_train += 1

        scheduler.step()
        print(scheduler.get_last_lr())
        epoch_loss_train = total_loss_train / num_batches_train

        model_descriptor.eval()
        model_scene.eval()

        # Evaluate validation sets
        with torch.no_grad():
            for j, (data_description, data_scene, data_img) in enumerate(val_loader):
                data_description = data_description.to(device)
                data_scene = data_scene.to(device)
                data_img = data_img.to(device)

                output_descriptor = model_descriptor(data_description)
                output_converter = model_converter(data_img)
                output_scene_ = model_scene(data_scene)
                output_scene = torch.cat((output_scene_, output_converter), dim=1)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)
                output_description_val[initial_index:final_index, :] = output_descriptor
                output_scene_val[initial_index:final_index, :] = output_scene

                multiplication = train_utility.cosine_sim(output_descriptor, output_scene)

                # define the target
                ground_truth = torch.eye(multiplication.size()[0])
                ground_truth = ground_truth.to(device)

                loss = contrastive_loss(multiplication, targets=ground_truth)

                total_loss_val += loss.item()

                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

        r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate(output_description=output_description_val,
                                                                  output_scene=output_scene_val,
                                                                  section="val")
        model_descriptor.train()
        model_scene.train()

        r10_hist.append(r10)
        if r10 > best_r10:
            best_r10 = r10
            train_utility.save_best_model(best_model_state_dict_scene=model_scene.state_dict(),
                                          best_model_state_dict_description=model_descriptor.state_dict(),
                                          best_model_state_dict_converter=model_converter.state_dict(),
                                          model_name=file_prefix + '.pt')

        print("train_loss:", epoch_loss_train)
        print("val_loss:", epoch_loss_val)

        train_losses.append(epoch_loss_train)
        val_losses.append(epoch_loss_val)

    # load best model for the evaluation stage
    best_model_state_dict_scene, best_model_state_dict_description, best_model_state_dict_converter = train_utility.load_best_model(file_prefix + '.pt')
    model_scene.load_state_dict(best_model_state_dict_scene)
    model_descriptor.load_state_dict(best_model_state_dict_description)
    model_converter.load_state_dict(best_model_state_dict_converter)

    # Evaluate Test set
    output_description_test = torch.empty(len(test_indices), output_feature_size).to(device)
    output_scene_test = torch.empty(len(test_indices), output_feature_size).to(device)
    with torch.no_grad():
        for j, (data_description, data_scene, data_img) in enumerate(test_loader):
            data_description = data_description.to(device)
            data_scene = data_scene.to(device)
            data_img = data_img.to(device)

            output_descriptor = model_descriptor(data_description)
            output_converter = model_converter(data_img)
            output_scene_ = model_scene(data_scene)
            output_scene = torch.cat((output_scene_, output_converter), dim=1)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_descriptor
            output_scene_test[initial_index:final_index, :] = output_scene

    r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate(output_description=output_description_test,
                                                              output_scene=output_scene_test,
                                                              section="test")
    print("best_r10", best_r10)


if __name__ == '__main__':
    start_train()
