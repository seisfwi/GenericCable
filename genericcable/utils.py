import numpy as np
from typing import Tuple

def get_length(traj: np.ndarray) -> float:
    """ Returns the length of the cable
    
    Parameters
    ----------
    traj : np.ndarray
        The trajectory of the cable. Shape (npts, 3)
    
    Returns
    -------
    float
        The length of the cable
    """

    if traj.shape[1] != 3 or traj.shape[0] < 2:
        raise ValueError("trajectory must be a 3D array with shape (npts, 3), "
                         "where npts >= 2")

    return np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)))


def select_along_traj(traj: np.ndarray, arc_len: float) -> np.ndarray:
    """Select the point along the trajectory of the das cable

    Parameters
    ----------
    traj : np.ndarray
        The trajectory of the cable. Shape (npts, 3)
    arc_len : float
        The arclength of the point of interest
    
    Returns
    -------
    np.ndarray
        The coordinates of the point of interest. Shape (3,)
    """

    # convert to np.float32
    traj = np.array(traj, dtype=np.float32)

    # check the input trajectory
    if traj.shape[1] != 3 or traj.shape[0] < 2:
        raise ValueError("trajectory must be a 3D array with shape (npts, 3), "
                         "where npts >= 2")

    # calculate the chord length of the das cable
    chord_lens = np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1))
    chord_len_total = np.sum(chord_lens)

    # check the input arclength
    if arc_len < 0 or arc_len > chord_len_total:
        raise ValueError(
            f"arclength must be within the range of [0, {chord_len_total}]")

    # find the index of the point of interest
    accum_len = 0.0
    for ip in range(traj.shape[0]):
        accum_len += chord_lens[ip]
        if accum_len >= arc_len:
            break

    # compute the desired point along the das cable by linear interpolation
    r = (accum_len - arc_len) / chord_lens[ip]
    l = 1.0 - r

    new_point = r * traj[ip, :] + l * traj[ip+1, :]

    # insert the new point into the trajectory
    new_point = np.array(new_point, dtype=np.float32)
    new_traj = np.insert(traj, ip+1, new_point, axis=0)
    new_traj = new_traj[ip+1:, :]

    return new_traj



def interparc(p: np.ndarray, interval: float, method: str = 'linear', 
              verbose: bool = False) -> np.ndarray:
    """ Interpolate points along a curve in 2 or more dimensions.

    Parameters
    ----------
    p : np.array
        The coordinates of the points defining the curve to be interpolated. 
        Must be of shape (npts, 3)
    interval : float
        The desired gauge length of the interpolated points
    method : str, optional
        The interpolation method to use. Either 'linear' or 'spline'.
        The default is 'linear'.
    verbose : bool, optional
        Print out the estimated number of points and the averaged error.

    Returns
    -------
    np.array
        The interpolated points. Shape (npts, 3)
    """

    # check method
    valid_methods = ['linear', 'spline']

    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
    elif method == 'spline':
        raise NotImplementedError(
            "Spline interpolation is not yet implemented")

    # check interval
    if interval <= 0:
        raise ValueError("interval must be positive")

    # check p
    npts = p.shape[0]
    ndim = p.shape[1]

    if ndim not in [2, 3]:
        raise ValueError("p must be 2D or 3D")

    if npts < 2:
        raise ValueError("p must have at least 2 points")

    # Calculate the approximated npts
    chordlen_all = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    chordlen = np.sum(chordlen_all)

    if chordlen <= interval:
        raise ValueError(f"Gauge length {interval} must be smaller than the "
                         "length of the curve {chordlen}")

    nt = int(np.floor(chordlen / interval))

    # adjust the last point to make sure the arc length is dividable by interval

    # length of interval * (nt-1)
    chordlen_desired = interval * nt

    # calculate the distance needed to be adjusted to the last point
    d = chordlen_desired - chordlen + chordlen_all[-1]

    # calculate the unit vector of the last segment
    u = (p[-1] - p[-2]) / chordlen_all[-1]

    # add the distance to the last point, assuming the last segment is straight
    p[-1] = p[-2] + d * u

    # calculate the new chord length
    chordlen_all = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    chordlen = np.sum(chordlen_all)

    nt = int(np.floor(chordlen / interval)) + 1

    # interpolate the curve with the desired number of points
    pt = interpolate_curve(nt, p, npts, ndim)

    if verbose:
        # calculate the error with the desired gauge length
        error = np.mean(
            abs(np.sqrt(np.sum(np.diff(pt, axis=0) ** 2, axis=1)) - interval))
        print(f"Desired interval: {interval}")
        print(f"Number of points: {nt}, with averaged error: {error:.6f}")

    # dudt = (p[tbins] - p[tbins-1]) / chordlen[tbins-1][:, np.newaxis]

    return pt


def interpolate_curve(nt: int, p: np.ndarray, npts: int, ndim: int) -> np.ndarray:
    """ Interpolate the curve with the desired number of points

    Parameters
    ----------
    nt : int
        The desired number of points
    p : np.array
        The coordinates of the points defining the curve to be interpolated.
        Must be of shape (npts, 3)
    npts : int
        The number of points in the original curve
    ndim : int
        The number of dimensions of the curve

    Returns
    -------
    np.array
        The interpolated points. Shape (npts, 3)
    """

    t = np.linspace(0, 1, nt)

    pt = np.full((nt, ndim), np.nan)
    chordlen = np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1))
    chordlen /= np.sum(chordlen)
    cumarc = np.hstack((0, np.cumsum(chordlen)))

    tbins = np.digitize(t, cumarc)
    tbins[(tbins <= 0) | (t <= 0)] = 1
    tbins[(tbins >= npts) | (t >= 1)] = npts - 1
    s = (t - cumarc[tbins-1]) / chordlen[tbins-1]
    pt = p[tbins-1] + (p[tbins] - p[tbins-1]) * s[:, np.newaxis]

    return pt


def frenet_serret(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Calculate the Frenet-Serret Space Curve Invariants
        _    r'
        T = ----  (Tangent)
            |r'|

        _    T'
        N = ----  (Normal)
            |T'|
        _   _   _
        B = T x N (Binormal)

        k = |T'|  (Curvature)

        t = dot(-B',N) (Torsion)

    Parameters
    ----------
    traj : np.array
        The trajectory of the curve. Must be of shape (npts, 3)
    
    Returns
    -------
    T : np.array
        The tangent vector. Shape (npts, 3)
    N : np.array
        The normal vector. Shape (npts, 3)
    B : np.array    
        The binormal vector. Shape (npts, 3)
    k : np.array
        The curvature. Shape (npts,)
    t : np.array
        The torsion. Shape (npts,)

    Notes
    -----
    The code below is benchmarked with MATLAB code: 
        Daniel Claxton (2023). Frenet (https://www.mathworks.com/matlabcentral/
        fileexchange/11169-frenet), MATLAB Central File Exchange. Retrieved 
        August 14, 2023.

    The Tanget is verified to be accurate, while the Normal and Binormal both 
    have some descrepancies, only at a few points. I check the reason and it is
    because of division of two very small numbers. I think it is fine to
    ignore these descrepancies.
    """

    # get the x, y, z coordinates of the trajectory
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    # Check input
    if x.shape != y.shape != z.shape:
        raise ValueError("x, y, z must have the same shape")

    # Convert to column vectors
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Speed of curve
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    dr = np.vstack((dx, dy, dz)).T

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    ddr = np.vstack((ddx, ddy, ddz)).T

    # Tangent
    T = dr / mag(dr, 3)

    # Derivative of tangent
    dTx = np.gradient(T[:, 0])
    dTy = np.gradient(T[:, 1])
    dTz = np.gradient(T[:, 2])
    dT = np.vstack((dTx, dTy, dTz)).T

    # Normal: the division of two very small numbers may have some descrepancies
    N = dT / mag(dT, 3)

    # Binormal
    B = np.cross(T, N)

    # Curvature
    k = mag(np.cross(dr, ddr), 1) / ((mag(dr, 1))**3)

    # Torsion
    t = -np.einsum('ij,ij->i', B, N)

    # Return the normal
    return T, N, B, k, t


def mag(T: np.ndarray, n: int) -> np.ndarray:
    """ Magnitude of a vector (Nx3)

    Parameters
    ----------
    T : np.array
        The vector. Shape (npts, 3)
    n : int
        The dimension of the vector
    
    Returns
    -------
    np.array
        The magnitude of the vector. Shape (npts,)
    """

    M = np.linalg.norm(T, axis=1)
    d = np.where(M == 0)[0]
    M[d] = np.finfo(float).eps * np.ones_like(d)
    M = M[:, np.newaxis]
    M = np.tile(M, (1, n))

    return M


def remove_duplicates(coords: np.ndarray, threshold: float) -> np.ndarray:
    """ Remove duplicate coordinates from the coords array

    Parameters
    ----------        
        coords (np.ndarray): 2D array of coordinates
        threshold (float): threshold distance to consider two points as the same

    Returns
    -------
        unique_coords (np.ndarray): 2D array of unique coordinates
    """

    unique_coords = []
    for coord in coords:
        is_duplicate = False
        for unique_coord in unique_coords:
            if np.linalg.norm(coord - unique_coord) <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_coords.append(coord)

    return np.array(unique_coords)


def build_lookup_table(coords: np.ndarray, unique_coords: np.ndarray) -> np.ndarray:
    """ Build a lookup table to map the coords to the unique_coords

    Parameters
    ----------
        coords (np.ndarray): 2D array of coordinates
        unique_coords (np.ndarray): 2D array of unique coordinates
    
    Returns
    -------
        lookup_table (np.ndarray): 1D array of indices
    """

    lookup_table = []
    for coord in coords:
        nearest_unique_idx = np.argmin(
            np.linalg.norm(unique_coords - coord, axis=1))
        lookup_table.append(nearest_unique_idx)

    return lookup_table



def smart_cable(beg, end, depth, well_interval, well_depth, well_width=0.1):
    """
    Generate a trajectory for a smart cable used in seismic modeling and inversion

    Parameters:
        beg (float): The starting position of the cable.
        end (float): The ending position of the cable.
        depth (float): The depth of the cable.
        well_interval (float): The interval between wells along the cable.
        well_depth (float): The depth of the wells.
        well_width (float, optional): The width of the wells. Defaults to 0.1.

    Returns:
        numpy.ndarray: An array containing the trajectory of the smart cable.

    Raises:
        AssertionError: If `beg` is greater than or equal to `end`.
        AssertionError: If `well_interval` is greater than or equal to the difference between `end` and `beg`.

    """

    assert beg < end, "must be increasing"
    assert well_interval < end - beg, "cable is not long enough even for one well"
    x = beg
    y = 0.0  # 2-D case

    traj = []

    # cable head
    traj.append([x, y, depth])

    while True:

        x = x + well_interval
        if x + well_width >= end:
            break

        # downgoing section of the well
        traj.append([x, y, depth])
        traj.append([x, y, depth + well_depth])

        # upgoing section of the well
        x += well_width
        traj.append([x, y, depth + well_depth])
        traj.append([x, y, depth])

    # cable end
    traj.append([end, y, depth])

    return np.array(traj)

