import numpy as np


def augment_polar_angles(images, code):
    def spatial(x):
        if code == 1:
            return np.rot90(x, 1).copy()
        if code == 2:
            return np.rot90(x, 2).copy()
        if code == 3:
            return np.rot90(x, 3).copy()
        if code == 4:
            return np.fliplr(x).copy()
        if code == 5:
            return np.flipud(x).copy()
        return x

    i0, i45, i90, i135 = [spatial(images[a]) for a in ("0", "45", "90", "135")]
    if code in (1, 3):
        i0, i90 = i90, i0
        i45, i135 = i135, i45
    elif code in (4, 5):
        i45, i135 = i135, i45
    return {"0": i0, "45": i45, "90": i90, "135": i135}
