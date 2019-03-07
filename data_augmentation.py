from imgaug import augmenters as iaa

# Returns a generator
# Usage batch_augmented = seq.augment_images(batch)
def get_seq(num_filters):

    """
    Main filters and augmentations for pilotnet data augmentation.
    """
    filters = iaa.SomeOf(num_filters, [
        iaa.ChangeColorspace("BGR"),
        iaa.ChangeColorspace("GRAY"),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 9)),
        iaa.MedianBlur(k=(3, 9)),
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.Add((-40, 40)),
        iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.Multiply((0.5, 1.5)),
        iaa.MultiplyElementwise((0.5, 1.5)),
        iaa.ContrastNormalization((0.5, 1.5)),
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
        iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.25),
        iaa.Sharpen(alpha=(0.6, 1.0)),
        iaa.Emboss(alpha=(0.0, 0.5)),
        iaa.CoarseDropout(0.2, size_percent=0.00001, per_channel = 1.0),
    ])
    affine = iaa.Affine(
        rotate=(-7, 7),
        scale=(0.9, 1.1),
        translate_percent=dict(x = (-0.05, 0.05)),
        mode = "symmetric",
    )

    return iaa.Sequential([
        filters,
        affine,
    ])