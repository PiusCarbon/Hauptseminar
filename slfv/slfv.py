import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt





############################################
# 1. Original update_allele_distribution
############################################

def _get_indices(z, r, length, discretization_steps):
    diff = r / (length / discretization_steps)
    
    lower_x = np.floor((z[0,:] * discretization_steps-diff) ).astype(int)
    upper_x = np.ceil((z[0,:] * discretization_steps+diff) ).astype(int)
    lower_x = np.maximum(lower_x, 0)
    upper_x = np.minimum(upper_x, discretization_steps-1)

    lower_y = np.floor((z[1,:] * discretization_steps-diff) ).astype(int)
    upper_y = np.ceil((z[1,:] * discretization_steps+diff) ).astype(int)
    lower_y = np.maximum(lower_y, 0)
    upper_y = np.minimum(upper_y, discretization_steps-1)
    return lower_x, upper_x, lower_y, upper_y


def _get_local_allele_distr_ball(allele_distr, z, r, no_alleles, discretization_steps, lower_x, upper_x, lower_y, upper_y):
    local_alleles = np.zeros(no_alleles)
    count = 0
    idx_set = []
    for i in range(lower_x, upper_x):
        for j in range(lower_y, upper_y):
            if (np.linalg.norm(z - (1/discretization_steps) * np.array([i, j]))) < r:
                local_alleles += allele_distr[i, j, :]
                count += 1
                idx_set.append((i, j))
    local_allele_distr = local_alleles / count
    return local_allele_distr, idx_set


def _get_parent_allele_ball(local_allele_distr, no_alleles):
    parent_allele_index = np.random.choice(no_alleles, p=local_allele_distr)
    parent_allele_array = np.zeros(no_alleles)
    parent_allele_array[parent_allele_index] = 1
    # Ensure the shapes match by expanding dimensions of parent_allele_array
    parent_allele_distr = parent_allele_array
    return parent_allele_distr


def _update_allele_distr_ball(allele_distr, u, parent_allele_distr, local_allele_distr, idx_set, t):
    for i, j in idx_set:
            allele_distr[i, j, :] = (1 - u[t]) * local_allele_distr + u[t] * parent_allele_distr


def ball_update_allele_distr(
        allele_distr: np.ndarray, 
        z: np.ndarray, 
        r: np.ndarray, 
        u: np.ndarray, 
        T: int, 
        no_alleles: int, 
        discretization_steps: int,
        length: int,
        plot=False
        ):
    
    lower_x, upper_x, lower_y, upper_y = _get_indices(z, r, length, discretization_steps)

    for t in range(T):
        local_allele_distr, idx_set = _get_local_allele_distr_ball(allele_distr, z[:, t], r[t], no_alleles, discretization_steps, lower_x[t], upper_x[t], lower_y[t], upper_y[t])

        # 2. sample a parent allele from the local distribution
        parent_allele_distr = _get_parent_allele_ball(local_allele_distr, no_alleles)
        
        # 3. Update the local distribution of alleles
        _update_allele_distr_ball(allele_distr, u, parent_allele_distr, local_allele_distr, idx_set, t)
    
    if plot:
        plt.imshow(allele_distr[:,:,0])
        plt.colorbar()
        plt.show()





############################################
# 2. Improved update_allele_distribution
############################################

def _get_local_allele_distr_rectangle(allele_distr, z, r, discretization_steps, lower_x, upper_x, lower_y, upper_y):
    local_alleles = allele_distr[lower_x:(upper_x + 1), lower_y:(upper_y + 1), :]
    local_allele_distr = np.mean(local_alleles, axis=(0,1))
    return local_allele_distr


def _get_parent_allele_rectangle(local_allele_distr, no_alleles):
    parent_allele_index = np.random.choice(no_alleles, p=local_allele_distr)
    parent_allele_array = np.zeros(no_alleles)
    parent_allele_array[parent_allele_index] = 1
    # Ensure the shapes match by expanding dimensions of parent_allele_array
    parent_allele_distr = parent_allele_array
    return parent_allele_distr


def _update_allele_distr_rectangle(allele_distr, u, parent_allele_distr, local_allele_distr, lower_x, upper_x, lower_y, upper_y, t):
    allele_distr[lower_x[t]:(upper_x[t] + 1), lower_y[t]:(upper_y[t] + 1),:] = (1 - u[t]) * local_allele_distr + u[t] * parent_allele_distr


def rectangle_update_allele_distribution(
        allele_distr: np.ndarray, 
        z: np.ndarray, 
        r: np.ndarray, 
        u: np.ndarray, 
        T: int, 
        no_alleles: int, 
        discretization_steps: int,
        length: int,
        plot=False
        ):
    
    lower_x, upper_x, lower_y, upper_y = _get_indices(z, r, length, discretization_steps)
    
    for t in range(T):
        # 1. get the local distribution of alleles at indices x,y
        local_allele_distr = _get_local_allele_distr_rectangle(allele_distr, z[:, t], r[t], discretization_steps, lower_x[t], upper_x[t], lower_y[t], upper_y[t])

        # 2. sample a parent allele from the local distribution
        parent_allele_distr = _get_parent_allele_rectangle(local_allele_distr, no_alleles)

        # 3. Update the local distribution of alleles
        _update_allele_distr_rectangle(allele_distr, u, parent_allele_distr, local_allele_distr, lower_x, upper_x, lower_y, upper_y, t)

    if plot:
        plt.imshow(allele_distr[:,:,0])
        plt.colorbar()
        plt.show()





############################################
# 3. Improved update_allele_distribution
############################################

def _get_neighbors_indices_impr(low_x, up_x, low_y, up_y, r, z, discretization_steps):
    '''
    Function to get the indices of neighboring grid points within a radius r of the event point z using a KD-tree
    '''
    x_coords = np.arange(low_x, up_x + 1)
    y_coords = np.arange(low_y, up_y + 1)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    radius = r * discretization_steps
    event_point = z * discretization_steps
    tree = cKDTree(grid_points)
    neighbors_indices = tree.query_ball_point(event_point, radius)
    return grid_points, neighbors_indices


# Function to compute the sum of allele_distr values for neighboring points
def _local_distr_impr(allele_distr, grid_points, neighbors_indices, no_alleles):
    sum_values = allele_distr[grid_points[neighbors_indices, 0] - 1, grid_points[neighbors_indices, 1] - 1, :].sum(axis=0)
    return sum_values / len(neighbors_indices)


def _get_parental_array_impr(local_distr_var, no_alleles):
    # 2. sample a parent allele from the local distribution
    assert np.isclose(np.sum(local_distr_var), 1), "Local distribution does not sum to 1 - {}".format(np.sum(local_distr_var))
    parent_allele_index = np.random.choice(no_alleles, p=local_distr_var)
    parent_allele_array = np.zeros(no_alleles)
    parent_allele_array[parent_allele_index] = 1
    return parent_allele_array


def _update_allele_distr_impr(grid_points, allele_distr, parent_allele_array, local_distr_var, u, t, neighbors_indices):
    for idx in neighbors_indices:
        allele_distr[grid_points[idx][0]-1, grid_points[idx][1]-1,:] = (1 - u[t]) * local_distr_var + u[t] * parent_allele_array


def improved_update_allele_distribution(
        allele_distr: np.ndarray, 
        z: np.ndarray, 
        r: np.ndarray, 
        u: np.ndarray, 
        T: int, 
        no_alleles: int, 
        discretization_steps: int, 
        length: int,
        plot=False
        ):
    
    # dictionary to store local grid and neighbors indices for reuse
    idx_dict = {}
    lower_x, upper_x, lower_y, upper_y = _get_indices(z, r, length, discretization_steps)

    for t in range(T):
        # check if index is already in dictionary
        key = (lower_x[t], upper_x[t], lower_y[t], upper_y[t])
        if key in idx_dict:
            local_grid, neighbors_indices = idx_dict[(lower_x[t], upper_x[t], lower_y[t], upper_y[t])]
        else:
            local_grid, neighbors_indices = _get_neighbors_indices_impr(lower_x[t], upper_x[t], lower_y[t], upper_y[t], r[t], z[:,t], discretization_steps)
            # write to dictionary for future use
            idx_dict[key] = (local_grid, neighbors_indices)


        if neighbors_indices != []:
            # 1. get the local distribution of alleles at indices x,y
            local_distr_var = _local_distr_impr(allele_distr, local_grid, neighbors_indices, no_alleles)
            
            # 2. sample a parent allele from the local distribution
            parent_allele_array = _get_parental_array_impr(local_distr_var, no_alleles)

            # 3. Update the local distribution of alleles
            _update_allele_distr_impr(local_grid, allele_distr, parent_allele_array, local_distr_var, u, t, neighbors_indices)
    
    if plot:
        plt.imshow(allele_distr[:,:,0])
        plt.colorbar()
        plt.show()