"""
GPU Manager - Centralized GPU management and optimization
"""

import torch
import gc


class GPUManager:
    """Centralized GPU management and optimization"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.gpu_memory_fraction = 0.8
        self.device = self._setup_gpu()
        self._setup_cuda_context()
        self._print_gpu_info()
    
    def _setup_gpu(self):
        """Setup optimal GPU device"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            return device
        else:
            print("[WARNING] CUDA not available, falling back to CPU")
            return torch.device('cpu')
    
    def _setup_cuda_context(self):
        """Setup CUDA context for optimal performance"""
        if self.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            dummy_tensor = torch.zeros(1, device=self.device)
            del dummy_tensor
            torch.cuda.empty_cache()
    
    def _print_gpu_info(self):
        """Print GPU information"""
        if self.cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[GPU] Using: {gpu_name}")
            print(f"[GPU] Total Memory: {gpu_memory:.1f} GB")
            print(f"[GPU] Memory Fraction: {self.gpu_memory_fraction*100}%")
        else:
            print("[GPU] CUDA not available - using CPU")
    
    def optimize_model(self, model):
        """Optimize model for GPU inference"""
        if self.cuda_available and hasattr(model, 'to'):
            model = model.to(self.device)
            if hasattr(model, 'half'):
                model = model.half()
        return model
    
    def cleanup_memory(self):
        """Cleanup GPU memory"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            gc.collect()


# Global GPU manager instance
gpu_manager = GPUManager()