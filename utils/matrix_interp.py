import torch

def se3_to_quaternion_translation(se3, tuple=True):
    q = matrix_to_quaternion(se3[..., :3, :3])
    t = se3[..., :3, 3]
    if tuple:
        return q, t
    else:
        return torch.cat((q, t), -1)


def quaternion_translation_to_se3(q: torch.Tensor, t: torch.Tensor):
    rmat = quaternion_to_matrix(q)
    rt4x4 = torch.cat((rmat, t[..., None]), -1)  # (..., 3, 4)
    rt4x4 = torch.cat((rt4x4, torch.zeros_like(rt4x4[..., :1, :])), -2)  # (..., 4, 4)
    rt4x4[..., 3, 3] = 1
    return rt4x4

def normalize(q, tol=1e-5):
    mag2 = (q * q).sum(dim=-1, keepdim=True)
    q = torch.where(mag2 > tol, q / (mag2 ** 0.5), q)
    return q

def matrix_interpolate(t0, t1, frac=0.5):
    tx_rot, tx_trans = se3_to_quaternion_translation(t0)
    ty_rot, ty_trans = se3_to_quaternion_translation(t1)

    with torch.no_grad():
        # SLERP
        interp_scalar = torch.zeros_like(tx_rot[..., 0:1])
        interp_scalar[...] = frac
        interp_rot = slerp(tx_rot, ty_rot, interp_scalar)
        interp_trans = (1 - interp_scalar) * tx_trans + interp_scalar * ty_trans
        tinterp = quaternion_translation_to_se3(interp_rot, interp_trans)

    return tinterp

def slerp(q0, q1, t):
    # See https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L847
    q0 = normalize(q0)
    q1 = normalize(q1)
    t = torch.clamp(t, min=0.0, max=1.0)

    cos = (q0 * q1).sum(dim=-1)

    # If the cos is negative, slerp won't take the shorter path.
    cos_neg_idx = cos < 0.0
    cos = torch.where(cos_neg_idx, -cos, cos)
    q0 = torch.where(cos_neg_idx.unsqueeze(-1), -q0, q0)

    cos = cos.unsqueeze(-1)

    theta_0 = torch.arccos(cos)
    sin_theta_0 = torch.sin(theta_0)

    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - cos * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    qr = torch.where(cos < 0.9995, s0 * q0 + s1 * q1, q0 + t * (q1 - q0))
    return normalize(qr)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        o: Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    q2 = quaternions**2
    rr, ii, jj, kk = torch.unbind(q2, -1)
    two_s = 2.0 / q2.sum(-1)
    ij = i * j
    ik = i * k
    ir = i * r
    jk = j * k
    jr = j * r
    kr = k * r

    o1 = 1 - two_s * (jj + kk)
    o2 = two_s * (ij - kr)
    o3 = two_s * (ik + jr)
    o4 = two_s * (ij + kr)

    o5 = 1 - two_s * (ii + kk)
    o6 = two_s * (jk - ir)
    o7 = two_s * (ik - jr)
    o8 = two_s * (jk + ir)
    o9 = 1 - two_s * (ii + jj)

    o = torch.stack((o1, o2, o3, o4, o5, o6, o7, o8, o9), -1)

    return o.view(quaternions.shape[:-1] + (3, 3))