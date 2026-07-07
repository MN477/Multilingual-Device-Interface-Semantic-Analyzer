"""
ppocr_inference.py
==================
Inference engine module for PP-OCRv6 text detection.
Uses native paddle.inference C++ runtime to run predictions.
"""

from pathlib import Path
import numpy as np

class PPOCRDetInference:
    """
    Direct model inference wrapper utilizing native paddle.inference API.
    Bypasses paddlex config overhead entirely.
    """
    def __init__(
        self,
        device: str = "cpu",
        model_dir: str = r"C:\Users\mouss\.paddlex\official_models\PP-OCRv6_medium_det",
    ):
        self.device = device
        self.model_dir = model_dir
        self._load_model()

    def _load_model(self):
        import paddle.inference as paddle_infer
        model_dir_path = Path(self.model_dir)
        model_file = model_dir_path / "inference.json"
        params_file = model_dir_path / "inference.pdiparams"

        if not model_file.exists() or not params_file.exists():
            raise FileNotFoundError(
                f"Model files not found at {self.model_dir}. "
                f"Make sure inference.json and inference.pdiparams are present."
            )

        config = paddle_infer.Config(str(model_file), str(params_file))

        if self.device.lower().startswith("gpu"):
            device_id = 0
            if ":" in self.device:
                device_id = int(self.device.split(":")[1])
            config.enable_use_gpu(100, device_id)
            config.exp_disable_mixed_precision_ops({"feed", "fetch"})
        else:
            config.disable_gpu()
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(10)

        # Optimize memory allocation
        config.enable_memory_optim()

        # Try to create predictor. Fallback if memory_optim is not supported
        try:
            self._predictor = paddle_infer.create_predictor(config)
        except Exception:
            # Recreate config without memory optimization
            config = paddle_infer.Config(str(model_file), str(params_file))
            if self.device.lower().startswith("gpu"):
                device_id = 0
                if ":" in self.device:
                    device_id = int(self.device.split(":")[1])
                config.enable_use_gpu(100, device_id)
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            else:
                config.disable_gpu()
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(10)
            self._predictor = paddle_infer.create_predictor(config)

        input_names = self._predictor.get_input_names()
        self._input_handle = self._predictor.get_input_handle(input_names[0])
        output_names = self._predictor.get_output_names()
        self._output_handle = self._predictor.get_output_handle(output_names[0])

    def __call__(self, batch_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference.
        Args:
            batch_tensor: (N, 3, H, W) float32 numpy array.
        Returns:
            pred_maps: Probability map output of shape (N, 1, H, W).
        """
        x = np.ascontiguousarray(batch_tensor, dtype=np.float32)
        self._input_handle.copy_from_cpu(x)
        self._predictor.run()
        pred_maps = self._output_handle.copy_to_cpu()
        return pred_maps
