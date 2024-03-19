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
from utils import normalize_percentile_to_255, add_contours_to_noise
from config import TranslatingConfig
from transform import *

def main(args):
    if args.by_volume:
        assert args.volume_specifier is not None
        assert args.slice_specifier is not None

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
        seed=args.seed,
        workers=args.workers,
        device=args.device,
        contour_channel_mode=args.contour_channel_mode,
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
        scheduler = DDPMScheduler(num_train_timesteps=args.denoise_step)
    elif args.model_type == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=args.denoise_step)

    ### Load the random noise seed generator
    generator = torch.manual_seed(config.seed)

    ### Create the directory to save the translated images
    save_directory = os.path.join(args.translating_folder_name)
    os.makedirs(save_directory, exist_ok=True)
    model.eval()

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
                    not_pass_flag = True
                    attempt = 0
                    mean_attemp = 0
                    img_buffer = []
                    mean_buffer = []
                    
                    while not_pass_flag:
                        ### Initialize the image as random noise
                        batch_size = args.num_copy
                        image_shape = (
                            batch_size,
                            config.in_channels,
                            config.img_size,
                            config.img_size,
                        )
                        image = randn_tensor(image_shape, generator=generator, device=model.device)
                        
                        ## Load contours
                        contour = Image.open(os.path.join(contour_directory, row["contour_name"]))
                        contour_tensor = val_transform_contour(contour)
                        contour_tensor = torch.unsqueeze(contour_tensor, dim=0)
    
                        scheduler.set_timesteps(config.denoise_step)
                        
                        ### Start denoising
                        progress_bar_denoise = tqdm(config.denoise_step)
                        for t in scheduler.timesteps:
                            image = add_contours_to_noise(image, contour_tensor, config, config.device, num_copy=args.num_copy, translation=True)
                            pred_noise = model(image, t).sample
    
                            ### Extract image channel
                            image = image[:, :args.in_channels, :, :]
    
                            ### Compute image at t-1
                            image = scheduler.step(pred_noise, t, image, generator=generator).prev_sample
    
                            progress_bar_denoise.update(1)
                            logs = {"denoised step": t}
                            progress_bar_denoise.set_postfix(**logs)
    
                        ### Process output and return the generated images
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

                        ### Iterate each generated input image, calculating the mean and the distance from previous generated slice if applicable
                        mean_list = []
                        dist_list = []
                        for img in image:
                            img = normalize_percentile_to_255(img)

                            ### Calculate mean
                            mean = np.mean(img)
                            mean_list.append(mean)
                            
                            ### Calculate distance from previous slice
                            if previous_slice is not None:
                                dist_list.append(calculate_Distance(img[:, :, 0], previous_slice))
                            else:
                                dist_list.append(0)

                            ### Check if any generated samples with mean value less than a specified value
                            ### Typically, smaller mean values imply qualitatively better samples
                            if mean < args.mean_threshold:
                                attempt += 1
                                not_pass_flag = False
                            else:
                                mean_attemp += 1
                                if mean_attemp >= args.max_attempt:
                                    print("--Warning--")
                                    print("Exceed maximum attempt! May need to use larger mean_threshold!")

                        ### If satisfying the mean requirement, then check the distance requirement if applicable
                        if not_pass_flag == False:
                            ### For the first slice, select the sample with the lowest mean value
                            if i == 0:
                                index = mean_list.index(min(mean_list))
                            else:
                                index = dist_list.index(min(dist_list))

                                ### If reaching the maximum attempts, then select the samples with smallest distance from all previous iterations
                                if attempt > args.max_attempt:
                                    index = dist_buffer.index(min(dist_buffer))
                                    img = img_buffer[index]
                                    img = Image.fromarray(img)
                                    img.save(os.path.join(save_directory, row["image_name"]))
                                    
                                    print("Reach maximum attempt")
                                    break

                                else:
                                    ### If the smallest distance is larger than a specified minimum value, then repeat the generation
                                    if min(dist_list) > args.dist_threshold:
                                        not_pass_flag = True

                                        img = image[index]
                                        img = np.squeeze(img)

                                        img_buffer.append(img)
                                        dist_buffer.append(dist_list[index])

                                        continue

                            ### Satisfying all the requirements, save the selected samples
                            previous_mean = mean_list[index]
                            img = image[index]
                            
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
        else:
            ### Translating images slice by slice (no need to be in the same volume)
            ### Spliting slices for parallel generation
            slice_list = df_val_meta["contour_name"].to_list()
            slice_split = np.array_split(slice_list, args.num_partition)
            slice_partition = slice_split[args.partition]

            df_by_slice = df_val_meta[df_val_meta["contour_name"].isin(slice_partition)]
            
            progress_bar_slice = tqdm(len(df_by_slice))
            for i, row in df_by_slice.iterrows():
                ### Initialize the image as random noise
                batch_size = args.num_copy
                image_shape = (
                    batch_size,
                    args.in_channels,
                    config.img_size,
                    config.img_size,
                )
                image = randn_tensor(image_shape, generator=generator, device=model.device)

                ### Load contours
                contour = Image.open(os.path.join(contour_directory, row["contour_name"]))
                contour_tensor = val_transform_contour(contour)
                contour_tensor = torch.unsqueeze(contour_tensor, dim=0)
                
                scheduler.set_timesteps(config.denoise_step)

                ### Start denoising
                progress_bar_denoise = tqdm(config.denoise_step)
                for t in scheduler.timesteps:
                    image = add_contours_to_noise(image, contour_tensor, config, config.device, num_copy=args.num_copy, translation=True)
                    pred_noise = model(image, t).sample

                    ## Extract image channel
                    image = image[:, :args.in_channels, :, :]

                    ## Compute image at t-1
                    image = scheduler.step(pred_noise, t, image, generator=generator).prev_sample

                    progress_bar_denoise.update(1)
                    logs = {"denoised step": t}
                    progress_bar_denoise.set_postfix(**logs)

                ### Process output and return the generated images
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().squeeze().numpy()

                ### Select the generated samples with the lowest mean value
                img_list = []
                mean_list = []
                for img in image:
                    img = normalize_percentile_to_255(img)

                    ### Calculate mean
                    mean = np.mean(img)
                    mean_list.append(mean)
                    img_list.append(img)
                    
                index = mean_list.index(min(mean_list))
                img = img_list[index]

                ### Save the generated samples
                img = Image.fromarray(img)
                img.save(os.path.join(save_directory, row["image_name"]))

                progress_bar_slice.update(1)
                logs = {"slice": i}
                progress_bar_slice.set_postfix(**logs)
            
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

    args = parser.parse_args()

    main(args)