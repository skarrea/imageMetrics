import numpy as np


def RMSRE(
    image_true: np.ndarray,
    image_test: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-9,
) -> float:
    """Root mean squared relative error (RMSRE) between two images within the
    specified mask. If not mask is specified the entire image is used.

    Parameters
    ----------
    image_true : np.ndarray
        ground truth image.
    image_test : np.ndarray
        predicted image.
    mask : np.ndarray, optional
        mask to compute the RMSRE in, by default None
    epsilon : float, optional
        epsilon used to stabilize the calculation of the relative error,
        by default 1e-9

    Returns
    -------
    float
        RMSRE value between the images within the specified mask.
    """
    if mask is None:
        mask = np.ones_like(image_true)

    mask_flat = mask.reshape(-1).astype(bool)

    # flatten
    relativeErrorImageFlat = (
        image_test.reshape(-1)[mask_flat] - image_true.reshape(-1)[mask_flat]
    ) / (image_true.reshape(-1)[mask_flat] + epsilon)

    return np.sqrt(
        np.mean(relativeErrorImageFlat) ** 2 + np.std(relativeErrorImageFlat) ** 2
    )


def RelativeErrorImage(
    image_true: np.ndarray,
    image_test: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Relative error image between two images within the specified mask. Values
    outside mask is set to zero. If no mask is specified the entire image is
    used.

    Parameters
    ----------
        image_true : np.ndarray
            ground truth image.
        image_test : np.ndarray
            predicted image.
        mask : np.ndarray, optional
            mask to compute the RMSRE in, by default None
        epsilon : float, optional
            epsilon used to stabilize the calculation of the relative error,
            by default 1e-9

    Returns
    -------
    np.ndarray
        Relative error image.
    """
    if mask is None:
        mask = np.ones_like(image_true)

    return np.where(mask, (image_test - image_true) / (image_true + epsilon), 0)


def MARE(
    image_true: np.ndarray,
    image_test: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-9,
) -> float:
    """Mean absolute relative error (MARE) between two images within the
    specified mask. If not mask is specified the entire image is used.

    Parameters
    ----------
    image_true : np.ndarray
        ground truth image.
    image_test : np.ndarray
        predicted image.
    mask : np.ndarray, optional
        mask to compute the RMSRE in, by default None
    epsilon : float, optional
        epsilon used to stabilize the calculation of the relative error,
        by default 1e-9

    Returns
    -------
    float
        MARE value between the images within the specified mask.
    """
    if mask is None:
        mask = np.ones_like(image_true)

    mask_flat = mask.reshape(-1).astype(bool)

    # flatten
    relativeErrorImageFlat = (
        image_test.reshape(-1)[mask_flat] - image_true.reshape(-1)[mask_flat]
    ) / (image_true.reshape(-1)[mask_flat] + epsilon)

    return np.mean(np.abs(relativeErrorImageFlat))

