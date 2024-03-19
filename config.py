from dataclasses import dataclass
from torchvision import transforms

@dataclass
class TrainingConfig:
    model_type: str = "ddpm"   # "ddpm" or "ddim"
    dataset: str = None
    input_domain: str = None
    output_domain: str = None
    img_size: int = 256
    in_channels: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 600
    gradient_accumulation_steps: int = 1
    noise_step: int = 1000
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 20
    mixed_precision: float = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None
    
    seed: int = 0
    workers: int = 8
    device: str = 'cuda:0'
    
    ## Augmentation setting
    scale_x = [0.8, 1.2]
    scale_y = [0.8, 1.2]
    degrees = 5
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = (0.9, 1.1)
    img_interpolation = transforms.InterpolationMode.BICUBIC
    contour_interpolation = transforms.InterpolationMode.NEAREST
    flip_p = 0.5
    apply_p = 0.9
    generator_seed: int = 42
    
    ## Eval config
    contour_guided: bool = False
    contour_channel_mode: str = "single"
    conditional: bool = False

@dataclass
class TranslatingConfig:
    model_type:str = "ddim"
    dataset: str = None
    input_domain: str = None
    output_domain: str = None
    eval_batch_size: int = 1
    img_size: int = 256
    denoise_step: int = 50
    selected_epoch: int = 1
    in_channels: int = 1
    output_dir: str = None

    seed: int = 0
    workers: int = 8
    device: str = 'cuda:0'

    by_volume: bool = False
    contour_channel_mode: str = "single"
    
    
    
    