import cv2
import numpy as np
from PIL import Image
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, input_tensor, class_index=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_index is None:
            class_index = int(torch.argmax(output, dim=1).item())

        score = output[:, class_index]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


def compute_brain_mask(original_image: np.ndarray) -> np.ndarray:
    """
    Compute a tight brain foreground mask using Otsu + morphology.
    Returns a float32 mask [0..1] of shape (H, W).
    """
    if original_image.dtype != np.uint8:
        original_image = original_image.astype(np.uint8)

    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If the mask covers more than 60% of pixels, invert (bright background)
    if np.mean(thresh > 0) > 0.6:
        thresh = cv2.bitwise_not(thresh)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    else:
        mask = opened

    # Slight dilation so we don't clip brain edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # Smooth mask edges for a natural blend
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    return mask.astype(np.float32) / 255.0


def generate_gradcam_from_tensor(
    input_tensor: torch.Tensor,
    original_image: np.ndarray,   # RGB numpy array, original unmasked image
    output_path: str,
    model,
    target_layer,
    class_index: int,
    device
):
    """
    Generate a brain-region-masked Grad-CAM overlay.
    Heatmap is constrained to the brain foreground; background stays dark/neutral.
    """
    model = model.to(device)
    model.eval()

    # ── 1. Compute Grad-CAM ──────────────────────────────────────────────────
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor, class_index=class_index)
    gradcam.remove_hooks()

    h, w = original_image.shape[:2]

    # ── 2. Resize CAM to original image size ─────────────────────────────────
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    # ── 3. Compute brain mask ────────────────────────────────────────────────
    brain_mask = compute_brain_mask(original_image)  # float32, [0..1], shape (H, W)

    # ── 4. Apply mask to CAM before colorising ───────────────────────────────
    cam_masked = cam_resized * brain_mask             # suppress background

    # Re-normalise so the colourmap uses the full range within the brain
    if cam_masked.max() > 1e-8:
        cam_masked = cam_masked / cam_masked.max()

    # ── 5. Build coloured heatmap ────────────────────────────────────────────
    heatmap_uint8 = np.uint8(255 * cam_masked)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    # ── 6. Convert original to BGR for OpenCV ───────────────────────────────
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # ── 7. Blend: inside brain = 55 % original + 45 % heatmap
    #            outside brain = original image (no heatmap noise) ───────────
    mask_3ch = np.stack([brain_mask] * 3, axis=-1)   # (H, W, 3)

    blended_brain = cv2.addWeighted(original_bgr, 0.55, heatmap_colored, 0.45, 0)

    # Outside the mask keep the original, weighted by inverse mask
    overlay = (blended_brain * mask_3ch + original_bgr * (1.0 - mask_3ch)).astype(np.uint8)

    # ── 8. Save ──────────────────────────────────────────────────────────────
    cv2.imwrite(output_path, overlay)