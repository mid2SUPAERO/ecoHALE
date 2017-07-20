def plot_3d_points(half_mesh, fname=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axes = []

    axes.append(fig.add_subplot(221, projection='3d'))
    axes.append(fig.add_subplot(222, projection='3d'))
    axes.append(fig.add_subplot(223, projection='3d'))
    axes.append(fig.add_subplot(224, projection='3d'))

    right_mesh = half_mesh.copy()
    right_mesh[:, :, 1] *= -1

    for i, ax in enumerate(axes):
        xs = half_mesh[:, :, 0]
        ys = half_mesh[:, :, 1]
        zs = half_mesh[:, :, 2]
        ax.plot_wireframe(xs, ys, zs, color='k')

        xs = right_mesh[:, :, 0]
        ys = right_mesh[:, :, 1]
        zs = right_mesh[:, :, 2]
        ax.plot_wireframe(xs, ys, zs, color='k')

        ax.set_xlim([20, 55])
        ax.set_ylim([-17.5, 17.5])
        ax.set_zlim([-17.5, 17.5])

        ax.set_axis_off()

        if i == 0:
            ax.view_init(elev=0, azim=0)
        elif i == 1:
            ax.view_init(elev=0, azim=90)
        elif i == 2:
            ax.view_init(elev=100000, azim=0)
        else:
            ax.view_init(elev=40, azim=-60)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    if fname:
        plt.savefig(fname + '.pdf')
    else:
        plt.show()

if __name__ == "__main__":

    import glob
    import os
    import numpy as np
    path = os.getcwd() + "/*.mesh.npy"
    for fname in glob.glob(path):
        mesh = np.load(fname)
        plot_3d_points(mesh, fname)
        print(fname)

        import subprocess
        pdf_name = fname.split('/')[-1]
        bash_string = 'pdfcrop --margins "0 0 0 0" ' + pdf_name + '.pdf ' + pdf_name.split('.')[0] + '.pdf'
        print(bash_string)
        subprocess.call(bash_string, shell=True)
        subprocess.call('rm ' + pdf_name + '.pdf', shell=True)
