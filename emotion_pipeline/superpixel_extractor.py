import numpy as np
import torch
from skimage.color import rgb2lab, rgb2hsv
from skimage.segmentation import slic
from sklearn.cluster import AgglomerativeClustering
from torchvision.transforms import functional as TF


class SuperpixelPaletteExtractor:
    """Extract palette and posterized image from superpixel clustering."""

    def __init__(self, n_segments=100, n_clusters=15, compactness=10.0):
        self.n_segments = n_segments
        self.n_clusters = n_clusters
        self.compactness = compactness

    def extract_palette(self, image_tensor: torch.Tensor):
        """
        image_tensor: torch (3,H,W) in [0,1]
        Returns:
          lab_palette (K,3), hsv_palette (K,3), rgb_palette (K,3),
          impacts (K,), merged_segments (H,W), posterized_rgb (H,W,3) in [0,1]
        """
        pil = TF.to_pil_image(image_tensor.cpu()).convert("RGB")
        rgb01 = np.asarray(pil).astype(np.float32) / 255.0

        image_lab = rgb2lab(rgb01)
        image_hsv = rgb2hsv(rgb01)

        segments = slic(
            rgb01,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0,
            channel_axis=-1,
            convert2lab=True,
            enforce_connectivity=True,
        )

        sp_labels = np.unique(segments)
        feats = np.vstack([image_lab[segments == sp].mean(axis=0) for sp in sp_labels])

        k = min(self.n_clusters, len(sp_labels))
        clustering = AgglomerativeClustering(n_clusters=k, linkage="average")
        sp_cluster = clustering.fit_predict(feats)

        sp2cl = np.zeros(int(sp_labels.max()) + 1, dtype=np.int32)
        for sp, cl in zip(sp_labels, sp_cluster):
            sp2cl[int(sp)] = int(cl)

        merged_segments = sp2cl[segments]

        final_labels = np.unique(merged_segments)
        h, w = merged_segments.shape
        total = h * w

        lab_palette, hsv_palette, rgb_palette, impacts = [], [], [], []
        for cl in final_labels:
            mask = merged_segments == cl
            lab_palette.append(image_lab[mask].mean(axis=0))
            hsv_palette.append(image_hsv[mask].mean(axis=0))
            rgb_palette.append(rgb01[mask].mean(axis=0))
            impacts.append(mask.sum() / total)

        lab_palette = torch.from_numpy(np.stack(lab_palette, 0)).float()
        hsv_palette = torch.from_numpy(np.stack(hsv_palette, 0)).float()
        rgb_palette = torch.from_numpy(np.stack(rgb_palette, 0)).float()
        impacts = torch.from_numpy(np.array(impacts, dtype=np.float32))

        rgb_pal_np = rgb_palette.numpy()
        posterized = rgb_pal_np[merged_segments]

        return lab_palette, hsv_palette, rgb_palette, impacts, merged_segments, posterized
