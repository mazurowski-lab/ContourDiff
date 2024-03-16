import torch

from diffusers.utils.torch_utils import randn_tensor 
from diffusers import DDPMPipeline, DDIMScheduler, DiffusionPipeline, ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from utils import make_grid, add_contours_to_noise

class ContourDiffDDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    
    model_cpu_offload_seq = "unet"
    
    def __init__(self, unet, scheduler, data_loader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.data_loader = data_loader
        self.external_config = external_config
        
    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        data_batch: Optional[torch.Tensor] = None,
        contour_batch: Optional[torch.Tensor] = None
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        ## Get the shape of the image (Not sure if needed)
        
        if self.external_config.contour_channel_mode == "single":
            img_channel = self.unet.config.in_channels - 1
        
        if self.external_config.eval_batch_size is not None:
            batch_size = self.external_config.eval_batch_size
        
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                img_channel,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, img_channel, *self.unet.config.sample_size)
        
        if self.external_config.conditional:
            ## Start froom source domain images, add noise up to certain steps, being there for denoising
            ## external_config.trans_noise_level: float [0-1]
            trans_start_t = int(self.external_config.trans_noise_level * (self.scheduler.config.num_train_timesteps - 1))

            clean_images = data_batch["images"].to(self.external_config.device)

            ## Sample noise, add to clean images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = torch.full(
                (clean_images.size(0),),
                trans_start_t,
                device=clean_images.device
            ).long()
            image = self.scheduler.add_noise(clean_images, noise, timesteps)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)
        
        ## Set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            if self.external_config.conditional:
                ## If t is larger than the customized noise adding steps, skip this iteration
                if t >= trans_start_t:
                    continue
            
            image = add_contours_to_noise(image, contour_batch, self.external_config, self.device)
                
            ## Predict the noise at t
            pred_noise = self.unet(image, t).sample
            
            ## Extract image channel
            image = image[:, :img_channel, :, :]
            
            ## Compute image at t-1
            image = self.scheduler.step(pred_noise, t, image, generator=generator).prev_sample
        
        ## Process output and return the generated images
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        
        if not return_dict:
            return (image,)
        
        return ImagePipelineOutput(images=image)

class ContourDiffDDIMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    model_cpu_offload_seq = "unet"
    
    def __init__(self, unet, scheduler, data_loader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.data_loader = data_loader
        self.external_config = external_config
        scheduler = DDIMScheduler.from_config(scheduler.config)
    
    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        data_batch: Optional[torch.Tensor] = None,
        contour_batch: Optional[torch.Tensor] = None,
        clean_img_type: str = "images"
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        ## Get the shape of the image (Not sure if needed)
        
        if self.external_config.contour_channel_mode == "single":
            img_channel = self.unet.config.in_channels - 1
        
        if self.external_config.eval_batch_size is not None:
            batch_size = self.external_config.eval_batch_size
        
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                img_channel,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, img_channel, *self.unet.config.sample_size)
            
        if self.external_config.conditional:
            ## Start froom source domain images, add noise up to certain steps, being there for denoising
            ## external_config.trans_noise_level: float [0-1]
            trans_start_t = int(self.external_config.trans_noise_level * (self.scheduler.config.num_train_timesteps - 1))

            clean_images = data_batch[clean_img_type].to(self.external_config.device)

            ## Sample noise, add to clean images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = torch.full(
                (clean_images.size(0),),
                trans_start_t,
                device=clean_images.device
            ).long()
            image = self.scheduler.add_noise(clean_images, noise, timesteps)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        ## Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if self.external_config.conditional:
                ## If t is larger than the customized noise adding steps, skip this iteration
                if t >= trans_start_t:
                    continue

            image = add_contours_to_noise(image, contour_batch, self.external_config, self.device)

            ## Predict the noise at t
            pred_noise = self.unet(image, t).sample

            ## Extract image channel
            image = image[:, :img_channel, :, :]

            ## Compute image at t-1
            image = self.scheduler.step(pred_noise, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator).prev_sample

        ## Process output and return the generated images
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
