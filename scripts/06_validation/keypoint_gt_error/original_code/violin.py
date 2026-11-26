import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def ksdensity_est(data, bandwidth=None, num_points=100):
    """
    Gaussian kernel density estimation similar to MATLAB ksdensity.
    
    Parameters:
      data: 1d array of data points (nan values are removed)
      bandwidth: (optional) scalar. If provided, passed to gaussian_kde as bw_method.
      num_points: number of grid points for evaluation
    
    Returns:
      f: estimated density evaluated at grid points (1d array)
      u: grid points (1d array)
      bb: effective bandwidth (scalar)
    """
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("No valid data points for KDE.")
    if bandwidth is not None:
        kde = gaussian_kde(data, bw_method=bandwidth)
    else:
        kde = gaussian_kde(data)
    u = np.linspace(np.min(data), np.max(data), num_points)
    f = kde(u)
    # effective bandwidth approximated by: kde.factor * standard deviation of data
    bb = kde.factor * np.std(data, ddof=1)
    return f, u, bb

def violin(Y, xlabel=None, facecolor=None, edgecolor='k', facealpha=0.5,
           mc='k', medc='r', bw=None, plotlegend=True, x=None):
    """
    Python version of MATLAB's violin plot based on ksdensity.
    
    INPUT:
      Y : Data to be plotted. Either an (n x m) numpy array (each column is a group)
          or a list/tuple where each element is a 1d array of data for one group.
      xlabel : Optional, list of string x-axis labels.
      facecolor: Face color for the violins. Default: [1, 0.5, 0]. Can be a single RGB tuple or
                 a list of such for each group.
      edgecolor: Edge (line) color. Default: 'k' (black). If set to None or '' or 'none',
                 mean/median lines are not plotted.
      facealpha: Transparency (default 0.5).
      mc: Color for mean line (default 'k').
      medc: Color for median line (default 'r').
      bw: Bandwidth. Either a scalar (applied to all groups) or an array/list with length equal to number of groups.
           If None, let gaussian_kde choose its default.
      plotlegend: If True, a legend for mean and median is added.
      x: Optional, x-axis positions for each group. If None, groups are placed at positions 1,2,...
    
    OUTPUT:
      h: list of matplotlib patch handles for the violins.
      L: legend handle (or None)
      MX: numpy array of means for each group.
      MED: numpy array of medians for each group.
      bws: numpy array of effective bandwidths used for each group.
    """
    # If Y is a 2D numpy array, convert to list of columns.
    if isinstance(Y, np.ndarray) and Y.ndim == 2:
        # Assume data organized as n x m
        Y_list = [Y[:, i] for i in range(Y.shape[1])]
    elif isinstance(Y, (list, tuple)):
        Y_list = Y
    else:
        raise ValueError("Y must be a 2D numpy array or a list/tuple of 1D arrays.")
        
    m = len(Y_list)  # number of groups
    
    # default facecolor
    if facecolor is None:
        fc_default = np.array([1, 0.5, 0])
    else:
        fc_default = np.array(facecolor)
    # If only one color is given, use the same color for every group
    if fc_default.ndim == 1:
        fc_array = np.tile(fc_default, (m, 1))
    else:
        fc_array = np.array(facecolor)
        if fc_array.shape[0] != m:
            raise ValueError("facecolor must be either a single RGB tuple or an array of shape (m, 3)")
    
    # Group positions on the x-axis
    if x is None:
        setX = False
        group_positions = np.arange(1, m+1)
    else:
        setX = True
        group_positions = np.asarray(x)
        if group_positions.size != m:
            raise ValueError("Length of x must equal the number of groups.")
    
    # Lists and arrays for results
    F_list = []
    U_list = []
    MX = np.zeros(m)
    MED = np.zeros(m)
    bws = np.zeros(m)
    
    # Run kernel density estimation for each group
    for i in range(m):
        data = np.array(Y_list[i]).flatten()  # 1d array
        # bw setting: if bw is given, scalar or list
        if bw is not None:
            if np.isscalar(bw):
                bw_i = bw
            else:
                bw = np.asarray(bw)
                if bw.size != m:
                    raise ValueError("bw must be scalar or an array with length equal to the number of groups")
                bw_i = bw[i]
        else:
            bw_i = None
        f, u, bb = ksdensity_est(data, bandwidth=bw_i, num_points=100)
        # Normalize and scale by 0.3 (same scaling as MATLAB)
        f = f / np.nanmax(f) * 0.3
        F_list.append(f)
        U_list.append(u)
        MX[i] = np.nanmean(data)
        MED[i] = np.nanmedian(data)
        bws[i] = bb

    # Start plotting
    fig = plt.gcf()
    ax = plt.gca()
    h = []         # patch handles
    p_mean = []    # mean line handles
    p_med = []     # median line handles
    
    # Draw each group
    for i in range(m):
        pos = group_positions[i]
        f = F_list[i]
        u = U_list[i]
        # Concatenate symmetrically left/right to build the polygon
        # right side: pos + f, lower side: pos - f (reversed)
        X_poly = np.concatenate([pos + f, (pos - f)[::-1]])
        Y_poly = np.concatenate([u, u[::-1]])
        patch = ax.fill(X_poly, Y_poly, color=fc_array[i], alpha=facealpha,
                          edgecolor=edgecolor if (edgecolor not in [None, '', 'none']) else 'none')
        h.append(patch[0])  # ax.fill returns a list of patch objects

        # Interpolate mean and median and draw horizontal lines
        # on the right curve (pos + f) and left curve (pos - f), respectively
        # Uses np.interp(x, xp, fp). MATLAB's interp1 requires xp to be ascending, so u should be ascending.
        # Mean
        if mc not in [None, '', 'none']:
            right_mean = np.interp(MX[i], u, pos + f)
            left_mean  = np.interp(MX[i], u[::-1], (pos - f)[::-1])
            line_mean, = ax.plot([left_mean, right_mean], [MX[i], MX[i]], color=mc, linewidth=2)
            p_mean.append(line_mean)
        # Median
        if medc not in [None, '', 'none']:
            right_med = np.interp(MED[i], u, pos + f)
            left_med  = np.interp(MED[i], u[::-1], (pos - f)[::-1])
            line_med, = ax.plot([left_med, right_med], [MED[i], MED[i]], color=medc, linewidth=2)
            p_med.append(line_med)
            
    # Legend settings
    L = None
    if plotlegend and ( (len(p_mean) > 0) or (len(p_med) > 0) ):
        handles = []
        labels = []
        if len(p_mean) > 0:
            handles.append(p_mean[0])
            labels.append('Mean')
        if len(p_med) > 0:
            handles.append(p_med[0])
            labels.append('Median')
        L = ax.legend(handles, labels, frameon=False, fontsize=14)
    
    # Axis settings (x-axis matches the number of groups)
    if not setX:
        ax.set_xlim([0.5, m+0.5])
    else:
        ax.set_xlim([np.min(group_positions) - 0.05*np.ptp(group_positions), 
                     np.max(group_positions) + 0.05*np.ptp(group_positions)])
    # y-axis uses the min/max of all u
    all_u = np.concatenate(U_list)
    ax.set_ylim([np.nanmin(all_u), np.nanmax(all_u)])
    
    # Set x-axis labels (if xlabel is given)
    if xlabel is not None:
        ax.set_xticks(group_positions)
        ax.set_xticklabels(xlabel)
    
    # Axis settings (tick length etc.)
    ax.tick_params(axis='both', which='both', length=0, labelsize=12)
    ax.grid(False)
    ax.set_box_aspect(1)
    
    # Returns the patch handles, legend, means, medians, and bandwidths
    return h, L, MX, MED, bws

# ---------------------------------
# Usage example below (same as the MATLAB example)
if __name__ == '__main__':
    # Generate random data as an example (4 groups)
    import scipy.stats as st
    np.random.seed(0)
    Y = np.column_stack([
        np.random.rand(1000),
        st.gamma.rvs(1, scale=2, size=1000),
        np.random.normal(10, 2, size=1000),
        st.gamma.rvs(10, scale=0.1, size=1000)
    ])
    
    # Example call
    fig, ax = plt.subplots(figsize=(8, 6))
    h, L, MX, MED, bws = violin(Y, xlabel=['a', 'b', 'c', 'd'], 
                                 facecolor=[[1,1,0], [0,1,0], [0.3,0.3,0.3], [0,0.3,0.1]],
                                 edgecolor='b', bw=0.3, mc='k', medc='r--')
    ax.set_ylabel(r'$\Delta [yesno^{-2}]$', fontsize=14)
    plt.show()
