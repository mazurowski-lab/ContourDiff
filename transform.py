from torchvision import transforms
from utils import NonUniformScaling

def load_train_transform_img(config):
    return transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size), interpolation=config.img_interpolation),
            transforms.RandomCrop(config.img_size, padding=32, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(p=config.flip_p),
            transforms.RandomApply([
                NonUniformScaling(scale_x_range=config.scale_x, scale_y_range=config.scale_y),
                transforms.RandomAffine(
                    degrees=config.degrees,
                    translate=config.translate,
                    scale=config.scale,
                    shear=config.shear
                ),
            ], p=config.apply_p),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def load_train_transform_contour(config):
    return transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size), interpolation=config.contour_interpolation),
            transforms.RandomCrop(config.img_size, padding=32, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(p=config.flip_p),
            transforms.RandomApply([
                NonUniformScaling(scale_x_range=config.scale_x, scale_y_range=config.scale_y),
                transforms.RandomAffine(
                    degrees=config.degrees,
                    translate=config.translate,
                    scale=config.scale,
                    shear=config.shear
                ),
            ], p=config.apply_p),
            transforms.Lambda(lambda x: x.point(lambda p: p > 50 and 255)),
            transforms.ToTensor(),
        ]
    )

def load_val_transform_img(config):
    return transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size), interpolation=config.img_interpolation),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def load_val_transform_contour(config):
    return transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size), interpolation=config.contour_interpolation),
            transforms.Lambda(lambda x: x.point(lambda p: p > 50 and 255)),
            transforms.ToTensor(),
        ]
    )

