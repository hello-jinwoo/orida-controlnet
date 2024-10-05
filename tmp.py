import torchvision.transforms as T
import torchvision.transforms.functional as F

# Create a composed transform for normalization
image_transforms = T.Compose(
    [
        # transforms.Resize(args.resolution, interpolation=T.InterpolationMode.BILINEAR),
        # transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Define the parameters for ColorJitter
color_jitter = T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3)

# Function to apply the same ColorJitter transform to both images
def apply_same_color_jitter(in_img, tgt_img):
    # Generate the random parameters for color jitter
    jitter_params = color_jitter.get_params(
        color_jitter.brightness, 
        color_jitter.contrast, 
        color_jitter.saturation, 
        color_jitter.hue
    )
    # Apply the same jitter parameters to both images
    in_img_jittered = F.adjust_brightness(in_img, jitter_params[0])
    in_img_jittered = F.adjust_contrast(in_img_jittered, jitter_params[1])
    in_img_jittered = F.adjust_saturation(in_img_jittered, jitter_params[2])
    in_img_jittered = F.adjust_hue(in_img_jittered, jitter_params[3])

    tgt_img_jittered = F.adjust_brightness(tgt_img, jitter_params[0])
    tgt_img_jittered = F.adjust_contrast(tgt_img_jittered, jitter_params[1])
    tgt_img_jittered = F.adjust_saturation(tgt_img_jittered, jitter_params[2])
    tgt_img_jittered = F.adjust_hue(tgt_img_jittered, jitter_params[3])

    return in_img_jittered, tgt_img_jittered

# Now apply transformations
in_img_jittered, tgt_img_jittered = apply_same_color_jitter(in_img, tgt_img)

# Apply the remaining transformations to both images
processed_examples["input_pixel_values"].append(image_transforms(in_img_jittered))
processed_examples["output_pixel_values"].append(image_transforms(tgt_img_jittered))