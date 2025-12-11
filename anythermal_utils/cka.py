import abc
from typing import Optional, Union

import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor


class CKABase(metaclass=abc.ABCMeta):
    def __init__(
        self, m: int, kernel: str, unbiased: bool = False, sigma: Optional[float] = 1.0
    ) -> None:
        """
        CKA abstract base class from which other CKA classes inherit.
        Args:
            m (int) - number of images / examples in a mini-batch or the full dataset;
            kernel (str) - 'linear' or 'rbf' kernel for computing the gram matrix;
            unbiased (bool) - whether to compute an unbiased version of CKA;
            sigma (float) - for 'rbf' kernel sigma defines the width of the Gaussian;
        """
        self.m = m  # number of examples
        self.kernel = kernel  # linear or rbf kernel
        self.unbiased = unbiased  # whether to use the unbiased version of CKA
        self.sigma = sigma  # width of the Gaussian in an rbf kernel

    @abc.abstractmethod
    def centering(self, K: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Centering of the (square) gram matrix K."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_kernel(self, X: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Compute the (square) gram matrix K."""
        raise NotImplementedError

    @abc.abstractmethod
    def linear_kernel(self, X: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Use a linear kernel for computing the gram matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def rbf_kernel(
        self, X: Union[Array, Tensor], sigma: float = None
    ) -> Union[Array, Tensor]:
        """Use an rbf kernel for computing the gram matrix. Sigma defines the width."""
        raise NotImplementedError

    @abc.abstractmethod
    def _hsic(
        self, X: Union[Array, Tensor], Y: Union[Array, Tensor]
    ) -> Union[Array, Tensor]:
        """Compute the Hilbert-Schmidt independence criterion."""
        raise NotImplementedError

    @abc.abstractmethod
    def compare(
        self, X: Union[Array, Tensor], Y: Union[Array, Tensor]
    ) -> Union[Array, Tensor]:
        """Compare two representation spaces X and Y using CKA."""
        raise NotImplementedError

import re
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

Array = np.ndarray

class CKANumPy(CKABase):
    def __init__(
        self, m: int, kernel: str, unbiased: bool = False, sigma: Optional[float] = 1.0
    ) -> None:
        """
        NumPy implementation of CKA.
        Args:
            m (int) - number of images / examples in a mini-batch or the full dataset;
            kernel (str) - 'linear' or 'rbf' kernel for computing the gram matrix;
            unbiased (bool) - whether to compute an unbiased version of CKA;
            sigma (float) - for 'rbf' kernel sigma defines the width of the Gaussian;
        """
        super().__init__(m=m, kernel=kernel, unbiased=unbiased, sigma=sigma)

    def centering(self, K: Array) -> Array:
        # The below code block is mainly copied from Simon Kornblith's implementation;
        # see here: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
        """Centering of the (square) gram matrix K."""
        if not np.allclose(K, K.T, rtol=1e-03, atol=1e-04):
            raise ValueError("\nInput array must be a symmetric matrix.\n")
        if self.unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = K.shape[0]
            np.fill_diagonal(K, 0.0)
            means = np.sum(K, axis=0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            K -= means[:, None]
            K -= means[None, :]
            np.fill_diagonal(K, 0.0)
        else:
            means = np.mean(K, axis=0, dtype=np.float64)
            means -= np.mean(means) / 2
            K -= means[:, None]
            K -= means[None, :]
        return K

    def apply_kernel(self, X: Array) -> Array:
        """Compute the (square) gram matrix K."""
        try:
            K = getattr(self, f"{self.kernel}_kernel")(X)
        except AttributeError:
            raise NotImplementedError
        return K

    def linear_kernel(self, X: Array) -> Array:
        """Use a linear kernel for computing the gram matrix."""
        return X @ X.T

    def rbf_kernel(self, X: Array) -> Array:
        """Use an rbf kernel for computing the gram matrix. Sigma defines the width."""
        GX = X @ X.T
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if self.sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        else:
            sigma = self.sigma
        KX *= -0.5 / sigma**2
        KX = np.exp(KX)
        return KX

    def _hsic(self, X: Array, Y: Array) -> Array:
        """Compute the Hilbert-Schmidt independence criterion."""
        K = self.apply_kernel(X)
        L = self.apply_kernel(Y)
        K_c = self.centering(K)
        L_c = self.centering(L)
        # np.sum(K_c * L_c) is equivalent to K_c.flatten() @ L_c.flatten() or in math
        # sum_{i=0}^{m} sum_{j=0}^{m} K^{\prime}_{ij} * L^{\prime}_{ij} = vec(K_c)^{T}vec(L_c)
        return np.sum(K_c * L_c)

    def compare(self, X: Array, Y: Array) -> Array:
        """Compare two representation spaces X and Y using CKA."""
        hsic_xy = self._hsic(X, Y)
        hsic_xx = self._hsic(X, X)
        hsic_yy = self._hsic(Y, Y)
        rho = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
        return rho
        


class CKATorch(CKABase):
    def __init__(
        self,
        m: int,
        kernel: str,
        unbiased: bool = False,
        device: str = "cpu",
        verbose: bool = False,
        sigma: Optional[float] = 1.0,
    ) -> None:
        """
        PyTorch implementation of CKA. Runs on CPU and CUDA.
        Args:
            m (int) - number of images / examples in a mini-batch or the full dataset;
            kernel (str) - 'linear' or 'rbf' kernel for computing the gram matrix;
            unbiased (bool) - whether to compute an unbiased version of CKA;
            device (str) - whether to perform CKA computation on CPU or GPU;
            sigma (float) - for 'rbf' kernel sigma defines the width of the Gaussian;
        """
        super().__init__(m=m, kernel=kernel, unbiased=unbiased, sigma=sigma)
        device = self._check_device(device, verbose)
        if device == "cpu":
            self.hsic = self._hsic
        else:
            # use JIT compilation on CUDA
            self.hsic = torch.compile(self._hsic)
        self.device = torch.device(device)

    @staticmethod
    def _check_device(device: str, verbose: bool) -> str:
        """Check whether the selected device is available on current compute node."""
        if device.startswith("cuda"):
            gpu_index = re.search(r"cuda:(\d+)", device)

            if not torch.cuda.is_available():
                warnings.warn(
                    "\nCUDA is not available on your system. Switching to device='cpu'.\n",
                    category=UserWarning,
                )
                device = "cpu"
            elif gpu_index and int(gpu_index.group(1)) >= torch.cuda.device_count():
                warnings.warn(
                    f"\nGPU index {gpu_index.group(1)} is out of range. "
                    f"Available GPUs: {torch.cuda.device_count()}. "
                    f"Switching to device='cuda:0'.\n",
                    category=UserWarning,
                )
                device = "cuda:0"
        if verbose:
            print(f"\nUsing device: {device}\n")
        return device

    def centering(self, K: TensorType["m", "m"]) -> TensorType["m", "m"]:
        """Centering of the (square) gram matrix K."""
        # The below code block is mainly copied from Simon Kornblith's implementation;
        # see here: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
        if not torch.allclose(K, K.T, rtol=1e-03, atol=1e-04):
            raise ValueError("\nInput array must be a symmetric matrix.\n")
        if self.unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = K.shape[0]
            K.fill_diagonal_(0.0)
            means = K.sum(dim=0) / (n - 2)
            means -= means.sum() / (2 * (n - 1))
            K -= means[:, None]
            K -= means[None, :]
            K.fill_diagonal_(0.0)
        else:
            means = K.mean(dim=0)
            means -= means.mean() / 2
            K -= means[:, None]
            K -= means[None, :]
        return K

    def apply_kernel(
        self, X: Union[TensorType["m", "d"], TensorType["m", "p"]]
    ) -> TensorType["m", "m"]:
        """Compute the (square) gram matrix K."""
        try:
            K = getattr(self, f"{self.kernel}_kernel")(X)
        except AttributeError:
            raise NotImplementedError
        return K

    def linear_kernel(
        self, X: Union[TensorType["m", "d"], TensorType["m", "p"]]
    ) -> TensorType["m", "m"]:
        """Use a linear kernel for computing the gram matrix."""
        return X @ X.T

    def rbf_kernel(
        self, X: Union[TensorType["m", "d"], TensorType["m", "p"]]
    ) -> TensorType["m", "m"]:
        """Use an rbf kernel for computing the gram matrix. Sigma defines the width."""
        GX = X @ X.T
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if self.sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
        else:
            sigma = self.sigma
        KX *= -0.5 / sigma**2
        KX = KX.exp()
        return KX

    def _hsic(
        self, X: TensorType["m", "d"], Y: TensorType["m", "p"]
    ) -> TensorType["1"]:
        K = self.apply_kernel(X)
        L = self.apply_kernel(Y)
        K_c = self.centering(K)
        L_c = self.centering(L)
        """Compute the Hilbert-Schmidt independence criterion."""
        # np.sum(K_c * L_c) is equivalent to K_c.flatten() @ L_c.flatten() or in math
        # sum_{i=0}^{m} sum_{j=0}^{m} K^{\prime}_{ij} * L^{\prime}_{ij} = vec(K_c)^{T}vec(L_c)
        return torch.sum(K_c * L_c)

    @staticmethod
    def _check_types(
        X: Union[Array, TensorType["m", "d"]],
        Y: Union[Array, TensorType["m", "p"]],
    ) -> Tuple[TensorType["m", "d"], TensorType["m", "p"]]:
        if isinstance(X, Array):
            X = torch.from_numpy(X)
        if isinstance(Y, Array):
            Y = torch.from_numpy(Y)
        return X, Y

    @torch.inference_mode()
    def compare(
        self,
        X: Union[Array, TensorType["m", "d"]],
        Y: Union[Array, TensorType["m", "p"]],
    ) -> TensorType["1"]:
        """Compare two representation spaces X and Y using CKA."""
        X, Y = self._check_types(X, Y)
        # move X and Y to current device
        X = X.to(self.device)
        Y = Y.to(self.device)
        hsic_xy = self.hsic(X, Y)
        hsic_xx = self.hsic(X, X)
        hsic_yy = self.hsic(Y, Y)
        rho = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        if rho.is_cuda:
            rho = rho.cpu()
        return rho

from typing import Optional, Union

BACKENDS = ["numpy", "torch"]


def get_cka(
    backend: str,
    m: int,
    kernel: str = "linear",
    unbiased: bool = False,
    sigma: Optional[float] = 1.0,
    device: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> Union[CKANumPy, CKATorch]:
    """Return a NumPy or PyTorch implementation of CKA."""
    assert backend in BACKENDS, f"\nSupported backends are: {BACKENDS}\n"
    if backend == "numpy":
        cka = CKANumPy(m=m, kernel=kernel, unbiased=unbiased, sigma=sigma)
    else:
        assert isinstance(
            device, str
        ), "\nDevice must be set for using PyTorch backend.\n"
        cka = CKATorch(
            m=m,
            kernel=kernel,
            unbiased=unbiased,
            device=device,
            sigma=sigma,
            verbose=verbose,
        )
    return cka