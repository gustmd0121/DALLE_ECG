import math
from math import sqrt
import argparse
from pathlib import Path
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# torch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch import DiscreteVAE
import ecg_plot
torch.manual_seed(0)
torch.set_num_threads(16)
# argument parsing

parser = argparse.ArgumentParser()

# parser.add_argument('--image_folder', type = str, required = True,
#                     help='path to your folder of images for learning the discrete VAE and its codebook')

parser.add_argument('--image_size', type = int, required = False, default = 5000,
                    help='image size')

parser = distributed_utils.wrap_arg_parser(parser)


train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--epochs', type = int, default = 1000, help = 'number of epochs')

train_group.add_argument('--batch_size', type = int, default = 256, help = 'batch size')

train_group.add_argument('--learning_rate', type = float, default = 2e-3, help = 'learning rate')

train_group.add_argument('--lr_decay_rate', type = float, default = 0.98, help = 'learning rate decay')

train_group.add_argument('--starting_temp', type = float, default = 1., help = 'starting temperature')

train_group.add_argument('--temp_min', type = float, default = 0.5, help = 'minimum temperature to anneal to')

train_group.add_argument('--anneal_rate', type = float, default = 1e-6, help = 'temperature annealing rate')

train_group.add_argument('--num_images_save', type = int, default = 1, help = 'number of images to save')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--num_tokens', type = int, default = 1024, help = 'number of ECG tokens')

model_group.add_argument('--num_layers', type = int, default = 4, help = 'number of layers (should be 3 or above)')

model_group.add_argument('--num_resnet_blocks', type = int, default = 2, help = 'number of residual net blocks')

model_group.add_argument('--smooth_l1_loss', dest = 'smooth_l1_loss', action = 'store_true')

model_group.add_argument('--emb_dim', type = int, default = 512, help = 'embedding dimension')

model_group.add_argument('--hidden_dim', type = int, default = 256, help = 'hidden dimension')

model_group.add_argument('--kl_loss_weight', type = float, default = 0., help = 'KL loss weight')

args = parser.parse_args()

# constants

IMAGE_SIZE = args.image_size
# IMAGE_PATH = args.image_folder

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LR_DECAY_RATE = args.lr_decay_rate

NUM_TOKENS = args.num_tokens
NUM_LAYERS = args.num_layers
NUM_RESNET_BLOCKS = args.num_resnet_blocks
SMOOTH_L1_LOSS = args.smooth_l1_loss
EMB_DIM = args.emb_dim
HIDDEN_DIM = args.hidden_dim
KL_LOSS_WEIGHT = args.kl_loss_weight

STARTING_TEMP = args.starting_temp
TEMP_MIN = args.temp_min
ANNEAL_RATE = args.anneal_rate

NUM_IMAGES_SAVE = args.num_images_save

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# data

# ds = ImageFolder(
#     IMAGE_PATH,
#     T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize(IMAGE_SIZE),
#         T.CenterCrop(IMAGE_SIZE),
#         T.ToTensor()
#     ])
# )
#
# if distributed_utils.using_backend(distributed_utils.HorovodBackend):
#     data_sampler = torch.utils.data.distributed.DistributedSampler(
#         ds, num_replicas=distr_backend.get_world_size(),
#         rank=distr_backend.get_rank())
# else:
#     data_sampler = None

###Added code####

new_original_files = torch.load("/home/hschung/ecg/lead_normalized_final.pt")


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = new_original_files

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


dataset = ECGDataset()

def normalize(dataset):
    channels_sum, channels_squared_sum = 0, 0
    for data in dataset:
        channels_sum += torch.mean(data, dim=1)
        channels_squared_sum += torch.mean(data**2, dim=1)

    mean = channels_sum / len(dataset)
    std = (channels_squared_sum / len(dataset) - mean**2)** 0.5

    mean = mean.unsqueeze(1)
    std = std.unsqueeze(1)

    normalized_data = []
    for data in dataset:
        data = (data - mean) / std
        normalized_data.append(data)
    return normalized_data

data = dataset.data
#normalized_data = normalize(dataset.data)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * train_size)
test_size = int(len(dataset) - (train_size + val_size))

training_data, validation_data, test_data = torch.utils.data.random_split(data, [train_size, val_size, test_size])

training_loader = DataLoader(dataset=training_data, batch_size=256, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size, shuffle=False)
Test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

dl = training_loader

###End added code###



# dl = DataLoader(ds, BATCH_SIZE, shuffle = not data_sampler, sampler=data_sampler)




vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HIDDEN_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    kl_div_loss_weight = KL_LOSS_WEIGHT
)
if not using_deepspeed:
    vae = vae.cuda()


# assert len(ds) > 0, 'folder does not contain any images'
# if distr_backend.is_root_worker():
#     print(f'{len(ds)} ECGs found for training')

# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)


if distr_backend.is_root_worker():
    # weights & biases experiment tracking

    import wandb

    model_config = dict(
        num_tokens = NUM_TOKENS,
        smooth_l1_loss = SMOOTH_L1_LOSS,
        num_resnet_blocks = NUM_RESNET_BLOCKS,
        kl_loss_weight = KL_LOSS_WEIGHT
    )

    run = wandb.init(
        project = 'dalle_train_vae',
        job_type = 'train_model',
        config = model_config
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {'train_batch_size': BATCH_SIZE}

(distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=args,
    model=vae,
    optimizer=opt,
    model_parameters=vae.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=sched if not using_deepspeed else None,
    config_params=deepspeed_config,
)

using_deepspeed_sched = False
# Prefer scheduler in `deepspeed_config`.
if distr_sched is None:
    distr_sched = sched
elif using_deepspeed:
    # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
    # handle its scheduling.
    using_deepspeed_sched = True

def save_model(path):
    save_obj = {
        'hparams': vae_params,
    }
    if using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + '-ds-cp'

        distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    save_obj = {
        **save_obj,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

# starting temperature

global_step = 0
temp = STARTING_TEMP

for epoch in range(EPOCHS):
    for i, data in enumerate(distr_dl):
        data = data.cuda()

        loss, recons = distr_vae(
            data,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        if using_deepspeed:
            # Gradients are automatically zeroed after the step
            distr_vae.backward(loss)
            distr_vae.step()
        else:
            distr_opt.zero_grad()
            loss.backward()
            distr_opt.step()

        logs = {}

        if i % 100 == 0:
            if distr_backend.is_root_worker():
                k = NUM_IMAGES_SAVE

                with torch.no_grad():
                    codes = vae.get_codebook_indices(data[:k])
                    hard_recons = vae.decode(codes)

                data, recons = map(lambda t: t[:k], (data, recons))
                # data, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (data, recons, hard_recons, codes))
                # data, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = False, range = (-1, 1)), (data, recons, hard_recons))
                data = data.squeeze(0)
                recons = recons.squeeze(0)
                ###ecg_plots###
                ecg_plot.plot(data.detach().cpu().numpy(), sample_rate=500, title = "Original ECG")
                ecg_plot.save_as_png("Original_ECG3")
                ecg_plot.plot(recons.detach().cpu().numpy(), sample_rate=500, title = "Reconstructed ECG")
                ecg_plot.save_as_png("Reconstructed_ECG3")
                ecg_plot.plot(hard_recons.detach().cpu().numpy(), sample_rate=500, title="Hard Reconstructed ECG")
                ecg_plot.save_as_png("Hard_Reconstructed_ECG3")



                logs = {
                    **logs,
                    'sample images':        wandb.log({"Original ECG": wandb.Image("Original_ECG3.png")}),
                    'reconstructions':      wandb.log({"Reconstructed ECG": wandb.Image("Reconstructed_ECG3.png")}),
                    'hard reconstructions': wandb.log({"Hard Reconstructed ECG": wandb.Image("Hard_Reconstructed_ECG3.png")}),
                    'codebook_indices':     wandb.Histogram(codes.cpu()),
                    'temperature':          temp
                }

                wandb.save('./vae/vae3.pt')
            torch.save(vae, './vae/vae3.pt')

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            # Do not advance schedulers from `deepspeed_config`.
            if not using_deepspeed_sched:
                distr_sched.step()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        if distr_backend.is_root_worker():
            if i % 10 == 0:
                lr = distr_sched.get_last_lr()[0]
                print(epoch, i, f'lr - {lr:6f} loss - {avg_loss.item()}')

                logs = {
                    **logs,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item(),
                    'lr': lr
                }

            wandb.log(logs)
        global_step += 1

    if distr_backend.is_root_worker():
        # save trained model to wandb as an artifact every epoch's end

        model_artifact = wandb.Artifact('trained-vae3', type = 'model', metadata = dict(model_config))
        model_artifact.add_file('./vae/vae3.pt')
        run.log_artifact(model_artifact)

if distr_backend.is_root_worker():
    # save final vae and cleanup

    torch.save(vae, './vae/vae-final3.pt')
    wandb.save('./vae/vae-final3.pt')

    model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
    model_artifact.add_file('./vae/vae-final3.pt')
    run.log_artifact(model_artifact)

    wandb.finish()
