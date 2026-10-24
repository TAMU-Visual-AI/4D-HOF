import cv2
import numpy as np

def preprocess_image(image_path: str, img_size: int = 224, mask=None,
                     apply_mask=False, full_image=True, rescale_factor=1.0) -> np.ndarray:
    """
    Mask and crop a frame before normalization, using a square hand/object box.

    Args:
        image_path: Path to the image file.
        img_size: Target image size (default 224).
        mask: Label mask (0 background, 1 object, 255 hand).
        apply_mask: Zero the background before cropping.
        full_image: Use the full frame instead of the mask's HOI bounding box.
        rescale_factor: Scale the padded square HOI crop.
    Returns:
        img_patch: Preprocessed image tensor of shape (3, img_size, img_size).
    """
    # 1. load image matching get_example
    cvimg = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % image_path)

    orig_height, orig_width = cvimg.shape[:2]

    if apply_mask or not full_image:
        if mask is None or mask.shape != cvimg.shape[:2]:
            raise ValueError(f"A mask matching the image shape is required for {image_path}")
    if apply_mask:
        cvimg[mask == 0] = 0

    # 2. generate the full-image or square HOI crop (empty masks use the full image)
    center_x, center_y = orig_width / 2.0, orig_height / 2.0
    bbox_width, bbox_height = orig_width, orig_height
    if not full_image:
        ys, xs = np.nonzero(mask)
        if len(xs):
            center_x = (xs.min() + xs.max() + 1) / 2.0
            center_y = (ys.min() + ys.max() + 1) / 2.0
            # Annotation HOI boxes also use 1.5x padding.
            bbox_width = bbox_height = max(xs.max() - xs.min() + 1,
                                           ys.max() - ys.min() + 1) * 1.5 * rescale_factor
    src = np.array([
        [center_x, center_y],
        [center_x, center_y + bbox_height / 2.0],
        [center_x + bbox_width / 2.0, center_y]
    ], dtype=np.float32)

    dst = np.array([
        [img_size / 2.0, img_size / 2.0],
        [img_size / 2.0, img_size],
        [img_size, img_size / 2.0]
    ], dtype=np.float32)

    trans = cv2.getAffineTransform(src, dst)

    # 3. crop and resize (equivalent to get_example logic)
    img_patch = cv2.warpAffine(cvimg, trans, (img_size, img_size),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    # convert from BGR to RGB
    img_patch = img_patch[:, :, ::-1]

    # 4. convert to float32 tensor (C, H, W)
    img_patch = img_patch.transpose(2, 0, 1).astype(np.float32)

    # 5. normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    for i in range(3):
        img_patch[i] = (img_patch[i] - mean[i]) / std[i]

    return img_patch
