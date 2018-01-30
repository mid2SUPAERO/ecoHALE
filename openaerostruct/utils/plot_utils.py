def plot_mesh_2d(ax, mesh, x_ind, y_ind, color):
    num_i, num_j, _ = mesh.shape

    for ind_i in range(num_i - 1):
        for ind_j in range(num_j):
            ax.plot(
                mesh[ind_i:ind_i + 2, ind_j, x_ind],
                mesh[ind_i:ind_i + 2, ind_j, y_ind],
                color=color,
            )
    for ind_i in range(num_i):
        for ind_j in range(num_j - 1):
            ax.plot(
                mesh[ind_i, ind_j:ind_j + 2, x_ind],
                mesh[ind_i, ind_j:ind_j + 2, y_ind],
                color=color,
            )


def scatter_2d(ax, array, x_ind, y_ind, color):
    ax.plot(array[:, x_ind], array[:, y_ind], 'o', color=color, markersize=3)


def arrow_2d(ax, array, d_array, x_ind, y_ind):
    for ind in range(array.shape[0]):
        ax.arrow(
            array[ind, x_ind], array[ind, y_ind],
            d_array[ind, x_ind], d_array[ind, y_ind],
            color='gray',
        )


# def plot_3d(mesh, color):
#     num_i, num_j, _ = mesh.shape
#
#     for ind_i in range(num_i - 1):
#         for ind_j in range(num_j):
#             ax.plot(
#                 mesh[ind_i:ind_i + 2, ind_j, 0],
#                 mesh[ind_i:ind_i + 2, ind_j, 1],
#                 mesh[ind_i:ind_i + 2, ind_j, 2],
#                 color
#             )
#     for ind_i in range(num_i):
#         for ind_j in range(num_j - 1):
#             ax.plot(
#                 mesh[ind_i, ind_j:ind_j + 2, 0],
#                 mesh[ind_i, ind_j:ind_j + 2, 1],
#                 mesh[ind_i, ind_j:ind_j + 2, 2],
#                 color
#             )
