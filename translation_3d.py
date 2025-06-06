import os
import torch
import argparse

## HF imports
from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

import pandas as pd
import numpy as np
from torch import nn
from tqdm.autonotebook import tqdm
from PIL import Image

## Custom imports
from utils import normalize_percentile_to_255
from config import TranslatingConfig
from transform import *

def main(args):
    if args.by_volume:
        assert args.volume_specifier is not None
        assert args.slice_specifier is not None

    def add_contours_to_noise_inference(noisy_images, contour, near_img, config, device, num_copy=1):
        if config.contour_channel_mode == "single":
            if num_copy > 1:
                contour = torch.cat([contour] * num_copy, dim=0)
            contour = contour.to(device)
            noisy_images = torch.cat((noisy_images, contour), dim=1)
        elif config.contour_channel_mode == "multi":
            if num_copy > 1:
                contour = torch.cat([contour] * num_copy, dim=0)
                near_img = torch.cat([near_img] * num_copy, dim=0)
            contour = contour.to(device)
            near_img = near_img.to(device)
            noisy_images = torch.cat((noisy_images, contour, near_img), dim=1)

        return noisy_images

    ### Load the translation config
    config = TranslatingConfig(
        model_type=args.model_type,
        dataset=args.dataset,
        img_size=args.img_size,
        input_domain=args.input_domain,
        output_domain=args.output_domain,
        in_channels=args.in_channels,
        eval_batch_size=args.eval_batch_size,
        denoise_step=args.denoise_step,
        training_noise_step=args.training_noise_step,
        seed=args.seed,
        workers=args.workers,
        device=args.device,
        contour_channel_mode=args.contour_channel_mode,
        near_guided=args.near_guided,
        num_copy=args.num_copy
    )
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f'ContourDiff-{config.input_domain}-{config.output_domain}-{config.model_type}-{config.dataset}'

    ### Load transform for images and contours
    val_transform_img = load_val_transform_img(config)
    val_transform_contour = load_val_transform_contour(config)

    ### Load the meta csv for translation
    ### Note: the code to generate df_translating_meta is not provided
    ### For translating by volume, you may need to add one column describing the volume specificier
    df_val_meta = pd.read_csv(os.path.join(args.data_directory, args.input_domain_meta_path), index_col=0)
    contour_directory = os.path.join(os.path.join(args.data_directory, args.input_domain_contour_folder))

    ### Load the checkpoints
    if args.selected_epoch is None:
        model_dir = os.path.join(config.output_dir, "model", "unet")
    else:
        model_dir = os.path.join(config.output_dir, f"model_epoch_{args.selected_epoch}", "unet")
    model = UNet2DModel.from_pretrained(model_dir, use_safetensors=True).to(config.device)

    if not args.by_volume:
        args.num_copy = 1

    ### Load the scheduler
    if args.model_type == "ddpm":
        scheduler = DDPMScheduler(num_train_timesteps=args.training_noise_step)
    elif args.model_type == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=args.training_noise_step)

    ### Load the random noise seed generator
    generator = torch.manual_seed(config.seed)

    ### Create the directory to save the translated images
    save_directory = os.path.join(args.translating_folder_name)
    os.makedirs(save_directory, exist_ok=True)
    model.eval()

    if config.near_guided:
        img_channel = 3

    with torch.no_grad():
        if args.by_volume:
            ### Translating images volume by volume
            ### Be sure to specify the 'volume_specifier' and 'slice_specifier'
            ### Spliting volumes for parallel generation
            unique_volume_list = np.sort(df_val_meta[args.volume_specifier].unique())
            unique_volume_split = np.array_split(unique_volume_list, args.num_partition)
            unique_volume_partition = unique_volume_split[args.partition]
            
            progress_bar_volume = tqdm(unique_volume_partition.shape[0])
            for n, volume in enumerate(unique_volume_partition):
                df_by_volume = df_val_meta[df_val_meta[args.volume_specifier] == volume]
                df_by_volume = df_by_volume.sort_values(by=args.slice_specifier).reset_index(drop=True)
                
                previous_slice = None

                progress_bar_slice = tqdm(len(df_by_volume))
                for i, row in df_by_volume.iterrows():
                    if i == 0:
                        batch_size = config.num_copy
                        image_shape = (
                            batch_size,
                            img_channel,
                            config.image_size,
                            config.image_size,
                        )
                        image = randn_tensor(image_shape, generator=generator, device=model.device)

                        ## Load contours
                        contour = Image.open(os.path.join(contour_directory, row["contour_name"]))
                        contour_tensor = val_transform_contour(contour)
                        contour_tensor = torch.unsqueeze(contour_tensor, dim=0)

                        near_img = Image.new('L', (contour_tensor.size(2), contour_tensor.size(3)), 0)
                        near_img_tensor = val_transform_img(near_img)
                        near_img_tensor = torch.unsqueeze(near_img_tensor, dim=0)

                        scheduler.set_timesteps(config.denoise_step)

                        ### Start denoising
                        progress_bar_denoise = tqdm(config.denoise_step)
                        for t in scheduler.timesteps:
                            image = add_contours_to_noise_inference(image, contour_tensor, near_img_tensor, config, config.device, num_copy=config.num_copy)
                            pred_noise = model(image, t).sample

                            ## Extract image channel
                            image = image[:, :img_channel, :, :]

                            ## Compute image at t-1
                            image = scheduler.step(pred_noise, t, image, generator=generator).prev_sample

                            progress_bar_denoise.update(1)
                            logs = {"denoised step": t}
                            progress_bar_denoise.set_postfix(**logs)

                        ## Process output and return the generated images
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

                        mean_list = []
                        std_list = []

                        for img in image:
                            img  = normalize_percentile_to_255(img)
                            mean = np.mean(np.array(img))
                            std = np.std(np.array(img))
                            mean_list.append(mean)
                            std_list.append(std)

                        index = mean_list.index(min(mean_list))
                        img = image[index]

                        img = np.squeeze(img)
                        img = normalize_percentile_to_255(img)
                        previous_slice = img

                        img = Image.fromarray(img)
                        img.save(os.path.join(save_directory, row["image_name"]))
                    else:
                        batch_size = 1
                        image_shape = (
                            batch_size,
                            img_channel,
                            config.image_size,
                            config.image_size,
                        )
                        image = randn_tensor(image_shape, generator=generator, device=model.device)

                        ## Load contours
                        contour = Image.open(os.path.join(contour_directory, row["contour_name"]))
                        contour_tensor = val_transform_contour(contour)
                        contour_tensor = torch.unsqueeze(contour_tensor, dim=0)

                        near_img = Image.fromarray(previous_slice)
                        near_img_tensor = val_transform_img(near_img)
                        near_img_tensor = torch.unsqueeze(near_img_tensor, dim=0)

                        scheduler.set_timesteps(config.denoise_step)

                        ### Start denoising
                        progress_bar_denoise = tqdm(config.denoise_step)
                        for t in scheduler.timesteps:
                            image = add_contours_to_noise_inference(image, contour_tensor, near_img_tensor, config, config.device, num_copy=1)
                            pred_noise = model(image, t).sample

                            ## Extract image channel
                            image = image[:, :img_channel, :, :]

                            ## Compute image at t-1
                            image = scheduler.step(pred_noise, t, image, generator=generator).prev_sample

                            progress_bar_denoise.update(1)
                            logs = {"denoised step": t}
                            progress_bar_denoise.set_postfix(**logs)

                        ## Process output and return the generated images
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

                        img = image[0]
                        img = np.squeeze(img)
                        img = normalize_percentile_to_255(img)
                        previous_slice = img

                        img = Image.fromarray(img)
                        img.save(os.path.join(save_directory, row["image_name"]))

                    progress_bar_slice.update(1)
                    logs = {"slice": i}
                    progress_bar_slice.set_postfix(**logs)
                    
                progress_bar_volume.update(1)
                logs = {"volume": n}
                progress_bar_volume.set_postfix(**logs)
            
    print("Finish translation!")
if __name__ == "__main__":
    # Parse args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help="name of the dataset")
    parser.add_argument('--img_size', type=int, default=256, help="size of the input images")
    parser.add_argument('--model_type', type=str, default="ddpm", choices=["ddpm", "ddim"], help="type of diffusion models (ddpm or ddim)")
    parser.add_argument('--eval_batch_size', type=int, default=1, help="validation batch size")
    parser.add_argument('--seed', type=int, default=0, help="seeds for random noise generator")
    parser.add_argument('--workers', type=int, default=0, help="number of workers")
    parser.add_argument('--device', type=str, default="cuda:0", help="gpu to use")
    parser.add_argument('--denoise_step', type=int, default=50, help="number of steps to denoise")
    parser.add_argument('--contour_channel_mode', type=str, default="single", help="number of channels for the contour")
    parser.add_argument('--output_dir', type=str, default=None, help="directory to save the output samples and checkpoints. If not specified, it will use the default name as ContourDiff-{input_domain}-{output_domain}-{model_type}-{dataset}")
    parser.add_argument('--translating_folder_name', type=str, default="translating_samples", help="name of the folder to save translated images")
    parser.add_argument('--selected_epoch', type=int, default=None, help="specifiy the epoch to load the checkpoints")
    parser.add_argument('--training_noise_step', type=int, default=1000, help="number of steps used for training the diffusion model")

    parser.add_argument('--data_directory', type=str, required=True, help="directory of the dataset")
    parser.add_argument('--input_domain', type=str, required=True, help="name of the input domain (e.g. CT, any)")
    parser.add_argument('--output_domain', type=str, required=True, help="name of the output domain (e.g. MRI)")
    parser.add_argument('--input_domain_contour_folder', type=str, required=True, help="name of the folder which contains the contours extract from the input domain to translate")
    parser.add_argument('--input_domain_meta_path', type=str, required=True, help="path of input domain meta under data_directory")
    parser.add_argument('--by_volume', action='store_true', help="specify if the translation is performed volume by volume")
    parser.add_argument('--volume_specifier', type=str, default=None, help="column name in the meta csv to contain volume information")
    parser.add_argument('--slice_specifier', type=str, default=None, help="column name in the meta csv to contain the slice information")
    parser.add_argument('--in_channels', type=int, default=1, help="number of channels for the input images")
    parser.add_argument('--num_copy', type=int, default=1, help="number of samples to generate per iteration")
    parser.add_argument('--mean_threshold', type=int, default=100, help="threshold for mean value requirement")
    parser.add_argument('--dist_threshold', type=int, default=50, help="threshold for distance requirement")
    parser.add_argument('--max_attempt', type=int, default=4, help="maximum number of iterations allowed for each slice")

    parser.add_argument("--num_partition", type=int, default=1, help="number of partitions to parallel")
    parser.add_argument("--partition", type=int, default=0, help="specify which partition to run")
    parser.add_argument("--near_guided", action="store_true", help="enable guidance from adjacent slice")
    parser.add_argument("--num_copy", type=int, default=1, help="number of candidates for intial slice")

    args = parser.parse_args()

    main(args)