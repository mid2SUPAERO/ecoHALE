""" Script to plot results from aero, struct, or aerostruct optimization.

Usage is
`python plot_all.py aero.db` for aero only,
`python plot_all.py struct.db` for struct only,
`python plot_all.py aerostruct.db` for aerostruct, or
`python plot_all.py __name__` for user-named database.

You can select a certain zoom factor for the 3d view by adding a number as a
last keyword.
The larger the number, the closer the view. Floats or ints are accepted.

Ex: `python plot_all.py aero.db 1` a wider view than `python plot_all.py aero.db 5`.

"""


from __future__ import division, print_function
import sys
major_python_version = sys.version_info[0]

if major_python_version == 2:
    import tkFont
    import Tkinter as Tk
else:
    import tkinter as Tk
    from tkinter import font as tkFont

from six import iteritems
import numpy as np
from openmdao.recorders.sqlite_reader import SqliteCaseReader

from read_all import read_hist, check_length

try:
    import matplotlib
    matplotlib.use('TkAgg')
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['axes.edgecolor'] = 'gray'
    matplotlib.rcParams['axes.linewidth'] = 0.5
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
        NavigationToolbar2TkAgg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.animation as manimation
    import sqlitedict
except:
    print()
    print("Correct plotting modules not available; please consult import list")
    print()

# TODO don't hardcode yield stress
yield_stress = 200e6

#####################
# User-set parameters
#####################

db_name = sys.argv[1]

# TODO change this for mission viz, for now jsut look at the first point
center = np.array([40., 5., 0.])

try:
    zoom_scale = sys.argv[3]
except:
    zoom_scale = 2.

class Display(object):
    def __init__(self, db_name):

        self.root = Tk.Tk()
        self.root.wm_title("Viewer")

        self.f = plt.figure(dpi=100, figsize=(12, 6), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.options_frame = Tk.Frame(self.root)
        self.options_frame.pack()

        toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.ax = plt.subplot2grid((10, 10), (0, 0), rowspan=10,
                                   colspan=5, projection='3d')

        self.num_iters = 0
        self.db_name = db_name
        self.show_wing = False
        self.show_tube = False
        self.show_mission = False
        self.curr_pos = 0
        self.old_n = 0
        self.aerostruct = False

        self.load_db()

        if self.show_wing and not self.show_tube:
            self.ax2 = plt.subplot2grid((10, 10), (0, 5), rowspan=5, colspan=5)
            self.ax3 = plt.subplot2grid((10, 10), (5, 5), rowspan=5, colspan=5)
        if self.show_tube and not self.show_wing:
            self.ax4 = plt.subplot2grid((10, 10), (0, 5), rowspan=5, colspan=5)
            self.ax5 = plt.subplot2grid((10, 10), (5, 5), rowspan=5, colspan=5)
        if self.show_wing and self.show_tube:
            self.ax2 = plt.subplot2grid((10, 10), (0, 5), rowspan=2, colspan=5)
            self.ax3 = plt.subplot2grid((10, 10), (2, 5), rowspan=2, colspan=5)
            self.ax4 = plt.subplot2grid((10, 10), (4, 5), rowspan=2, colspan=5)
            self.ax5 = plt.subplot2grid((10, 10), (6, 5), rowspan=2, colspan=5)
        if self.show_mission:
            self.ax6 = plt.subplot2grid((10, 10), (8, 5), rowspan=2, colspan=5)

    def load_db(self):

        self.data_all_iters, self.show_wing, self.show_tube, self.show_mission = read_hist(self.db_name)

        self.num_iters = len(self.data_all_iters)

        if self.show_wing:

            data = self.data_all_iters[0]
            num_nodes = data['mesh'].shape[0]
            self.pt_list = range(num_nodes)

            for data in self.data_all_iters:
                data['lift'] = []
                data['lift_ell'] = []

                for pt in self.pt_list:
                    m_vals = data['mesh'][pt]
                    cvec = m_vals[0, :, :] - m_vals[-1, :, :]
                    chords = np.sqrt(np.sum(cvec**2, axis=1))
                    chords = 0.5 * (chords[1:] + chords[:-1])
                    a = data['alpha_rad']
                    cosa = np.cos(a)
                    sina = np.sin(a)

                    widths = m_vals[0, 1:, 2] - m_vals[0, :-1, 2]

                    forces = data['forces'][pt]
                    rho_kg_m3 = data['rho_kg_m3'][pt]
                    v_m_s = data['v_m_s'][pt]

                    lift = forces[:, 1] / widths/0.5/rho_kg_m3/v_m_s**2

                    span = (m_vals[0, :, 2] / (m_vals[0, -1, 2] - m_vals[0, 0, 2]))
                    span = span - (span[0] + .5)

                    lift_area = np.sum(lift * (span[1:] - span[:-1]))
                    lift /= lift_area

                    lift_ell = 4  / np.pi * np.sqrt(1 - (2*span)**2)

                    data['lift'].append(lift)
                    data['lift_ell'].append(lift_ell)

                data['mesh'] -= center
                data['fea_mesh'] -= center


    def plot_sides(self):

        if self.show_wing:

            self.ax2.cla()
            self.ax2.locator_params(axis='y',nbins=5)
            self.ax2.locator_params(axis='x',nbins=3)
            self.ax2.set_ylim([-10., 11.])
            self.ax2.set_xlim([-1, 1])
            self.ax2.set_ylabel('twist', rotation="horizontal", ha="right")

            self.ax3.cla()
            self.ax3.text(0.05, 0.7, 'elliptical',
                transform=self.ax3.transAxes, color='g')
            self.ax3.locator_params(axis='y',nbins=4)
            self.ax3.locator_params(axis='x',nbins=3)
            self.ax3.set_ylim([0, .25])
            self.ax3.set_xlim([-1, 1])
            self.ax3.set_ylabel('lift', rotation="horizontal", ha="right")

        if self.show_tube:

            # thickness plot!
            self.ax4.cla()
            self.ax4.locator_params(axis='y',nbins=4)
            self.ax4.locator_params(axis='x',nbins=3)
            # TODO change thickness bounds
            self.ax4.set_ylim([0., .2])
            self.ax4.set_xlim([-1, 1])
            self.ax4.set_ylabel('thickness', rotation="horizontal", ha="right")

            # CL plot!
            # self.ax4.cla()
            # self.ax4.locator_params(axis='y',nbins=4)
            # self.ax4.locator_params(axis='x',nbins=3)
            # # TODO change thickness bounds
            # self.ax4.set_ylim([0., .9])
            # self.ax4.set_ylabel('CL', rotation="horizontal", ha="right")

            self.ax5.cla()
            self.ax5.axhline(yield_stress, c='r', lw=2, ls='--')

            self.ax5.locator_params(axis='y',nbins=4)
            self.ax5.locator_params(axis='x',nbins=3)
            self.ax5.set_ylim([0, yield_stress*1.1])
            self.ax5.set_xlim([-1, 1])
            self.ax5.set_ylabel('von mises', rotation="horizontal", ha="right")
            self.ax5.text(0.075, 1.1, 'failure limit',
                transform=self.ax5.transAxes, color='r')

        if self.show_mission:
            self.ax6.cla()
            self.ax6.set_ylim([0., 15.])
            self.ax6.locator_params(axis='y',nbins=4)
            self.ax6.locator_params(axis='x',nbins=5)
            self.ax6.set_ylabel('altitude', rotation='horizontal', ha='right')

        data = self.data_all_iters[self.curr_pos]

        if self.show_tube:
            for i, pt in enumerate(self.pt_list):
                fea_mesh = data['fea_mesh'][pt, :, :]
                span = fea_mesh[-1, 2] - fea_mesh[0, 2]
                rel_span = (fea_mesh[:, 2] - fea_mesh[0, 2]) * 2 / span - 1
                span_diff = ((fea_mesh[:-1, 2] + fea_mesh[1:, 2]) / 2 - fea_mesh[0, 2]) * 2 / span - 1

                thick_vals = data['thickness'][pt]
                # TODO: check ths out for multiple node case; will need to reformulate since OM flattens constraints
                vm_vals = data['vonmises'][pt] * yield_stress
                if pt==0:
                    print(thick_vals)
                    self.ax4.plot(span_diff, thick_vals, lw=2, c='b')

                self.ax5.plot(span_diff, vm_vals, lw=2, c=cm.viridis(i/len(self.pt_list)))

        if self.show_wing:
            for i, pt in enumerate(self.pt_list):
                mesh = data['mesh'][pt, 0, :, :]
                span = mesh[-1, 2] - mesh[0, 2]
                rel_span = (mesh[:, 2] - mesh[0, 2]) * 2 / span - 1
                span_diff = ((mesh[:-1, 2] + mesh[1:, 2]) / 2 - mesh[0, 2]) * 2 / span - 1

                twist = data['twist'][pt] * 180. / np.pi
                lift = data['lift'][pt]
                lift_ell = data['lift_ell'][pt]

                if pt==0:
                    self.ax3.plot(rel_span, lift_ell, '--', lw=2, c='g')
                self.ax2.plot(rel_span, twist, lw=2, c=cm.viridis(i/len(self.pt_list)))
                self.ax3.plot(span_diff, lift, lw=2, c=cm.viridis(i/len(self.pt_list)))
                self.ax3.set_ylim([0., 2.])

        if self.show_mission:
            # self.ax4.plot(data['x_1e3_km'], data['C_L'], c='b')
            self.ax6.plot(data['x_1e3_km'], data['h_km'], c='b')

    def plot_wing(self):

        self.ax.cla()
        az = self.ax.azim
        el = self.ax.elev
        dist = self.ax.dist

        data = self.data_all_iters[self.curr_pos]

        self.ax.set_axis_off()

        pt = self.pt_list[0]

        if self.show_wing:

            # TODO: change this to deformed mesh if aerostructural
            mesh = data['mesh'][pt]

            # If aerostructural, show the deformed fea mesh
            if self.show_tube:
                mesh = mesh + data['disp'][pt, :, :3]

            self.ax.set_axis_off()

            x = mesh[:, :, 0]
            y = mesh[:, :, 1]
            z = mesh[:, :, 2]

            self.ax.plot_wireframe(x, z, y, rstride=1, cstride=1, color='k')

        if self.show_tube:
            fea_mesh = data['fea_mesh'][pt, :, :]

            if self.show_wing:
                fea_mesh = fea_mesh + data['disp'][pt, :, :3]

            # Get the array of radii and thickness values for the FEM system
            r0 = data['radius'][pt]
            t0 = data['thickness'][pt]

            # Create a normalized array of values for the colormap
            colors = t0
            colors = colors / np.max(colors)

            # Set the number of rectangular patches on the cylinder
            num_circ = 12
            # TODO get fem_origin info
            fem_origin = 0.35

            # Get the number of spanwise nodal points
            n = fea_mesh.shape[1]

            # Create an array of angles around a circle
            p = np.linspace(0, 2*np.pi, num_circ)

            # Loop through each element in the FEM system
            for i, thick in enumerate(t0):

                # Get the radii describing the circles at each nodal point
                r = np.array((r0[i], r0[i]))
                R, P = np.meshgrid(r, p)

                # Get the X and Z coordinates for all points around the circle
                X, Z = R*np.cos(P), R*np.sin(P)

                # Add the location of the element centers to the circle coordinates
                X[:, 0] += fea_mesh[i, 0]
                X[:, 1] += fea_mesh[i+1, 0]
                Z[:, 0] += fem_origin * (fea_mesh[i, 1] - fea_mesh[i, 1]) + fea_mesh[i, 1]
                Z[:, 1] += fem_origin * (fea_mesh[i+1, 1] - fea_mesh[i+1, 1]) + fea_mesh[i+1, 1]

                # Get the spanwise locations of the spar points
                Y = np.empty(X.shape)
                Y[:] = np.linspace(fea_mesh[i, 2], fea_mesh[i+1, 2], 2)

                # Set the colors of the rectangular surfaces
                col = np.zeros(X.shape)
                col[:] = colors[i]

                # Plot the rectangular surfaces for each individual FEM element
                try:
                    self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        facecolors=cm.viridis(col), linewidth=0)
                except:
                    self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        facecolors=cm.coolwarm(col), linewidth=0)

        round_to_n = lambda x, n: round(x[0], -int(np.floor(np.log10(abs(x[0])))) + (n - 1))
        obj_val = round_to_n(data['obj'] * 1e6 / 9.81, 7)
        self.ax.text2D(.55, .05, 'Fuel burn: {} kg'.format(obj_val),
        transform=self.ax.transAxes, color='k')

        # TODO: move this elsewhere
        lim = 0.
        if self.show_wing:
            for data in self.data_all_iters:
                lim = max(np.max(data['mesh'][pt], axis=(0,1,2)), lim)
            lim /= float(zoom_scale)
        else:
            for data in self.data_all_iters:
                lim = max(np.max(data['fea_mesh'][pt], axis=(0,1)), lim)
            lim /= float(zoom_scale)
        self.ax.auto_scale_xyz([-lim, lim], [-lim, lim], [-lim, lim])
        self.ax.set_title("Major Iteration: {}".format(self.curr_pos))

        self.ax.view_init(elev=el, azim=az)  # Reproduce view
        self.ax.dist = dist

    def save_video(self):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie', artist='Matplotlib')
        writer = FFMpegWriter(fps=5, metadata=metadata, bitrate=3000)

        with writer.saving(self.f, "movie.mp4", 100):
            self.curr_pos = 0
            self.update_graphs()
            self.f.canvas.draw()
            plt.draw()
            for i in range(10):
                writer.grab_frame()

            for i in range(self.num_iters):
                self.curr_pos = i
                self.update_graphs()
                self.f.canvas.draw()
                plt.draw()
                writer.grab_frame()

            self.curr_pos = self.num_iters
            self.update_graphs()
            self.f.canvas.draw()
            plt.draw()
            for i in range(20):
                writer.grab_frame()

    def update_graphs(self, e=None):
        if e is not None:
            self.curr_pos = int(e)
            self.curr_pos = self.curr_pos % (self.num_iters + 1)

        self.plot_sides()
        self.plot_wing()
        self.canvas.show()

    def check_length(self):
        self.num_iters = check_length(self.db_name)

    def get_list_limits(self, input_list):
        list_min = 1.e20
        list_max = -1.e20
        for list_ in input_list:
            mi = np.min(list_)
            if mi < list_min:
                list_min = mi
            ma = np.max(list_)
            if ma > list_max:
                list_max = ma

        return list_min, list_max


    def auto_ref(self):
        """
        Automatically refreshes the history file, which is
        useful if examining a running optimization.
        """
        if self.var_ref.get():
            self.root.after(500, self.auto_ref)
            self.check_length()
            self.update_graphs()

            # Check if the sqlitedict file has change and if so, fully
            # load in the new file.
            if self.num_iters > self.old_n:
                self.load_db()
                self.old_n = self.num_iters
                self.draw_slider()

    def save_image(self):
        fname = 'fig' + '.pdf'
        plt.savefig(fname)

    def quit(self):
        """
        Destroy GUI window cleanly if quit button pressed.
        """
        self.root.quit()
        self.root.destroy()

    def draw_slider(self):
        # scale to choose iteration to view
        self.w = Tk.Scale(
            self.options_frame,
            from_=0, to=self.num_iters-1,
            orient=Tk.HORIZONTAL,
            resolution=1,
            font=tkFont.Font(family="Helvetica", size=10),
            command=self.update_graphs,
            length=200)

        if self.curr_pos == self.num_iters - 1 or self.curr_pos == 0 or self.var_ref.get():
            self.curr_pos = self.num_iters
        self.w.set(self.curr_pos)
        self.w.grid(row=0, column=1, padx=5, sticky=Tk.W)

    def draw_GUI(self):
        """
        Create the frames and widgets in the bottom section of the canvas.
        """
        font = tkFont.Font(family="Helvetica", size=10)

        lab_font = Tk.Label(
            self.options_frame,
            text="Iteration number:",
            font=font)
        lab_font.grid(row=0, column=0, sticky=Tk.S)

        self.draw_slider()

        # Option to automatically refresh history file
        # especially useful for currently running optimizations
        self.var_ref = Tk.IntVar()
        # self.var_ref.set(1)
        c11 = Tk.Checkbutton(
            self.options_frame,
            text="Automatically refresh",
            variable=self.var_ref,
            command=self.auto_ref,
            font=font)
        c11.grid(row=0, column=4, sticky=Tk.W, pady=6)

        button = Tk.Button(
            self.options_frame,
            text='Save video',
            command=self.save_video,
            font=font)
        button.grid(row=0, column=5, padx=5, sticky=Tk.W)

        button4 = Tk.Button(
            self.options_frame,
            text='Save image',
            command=self.save_image,
            font=font)
        button4.grid(row=0, column=6, padx=5, sticky=Tk.W)

        button5 = Tk.Button(
            self.options_frame,
            text='Quit',
            command=self.quit,
            font=font)
        button5.grid(row=0, column=7, padx=5, sticky=Tk.W)

        self.auto_ref()

def disp_plot(db_name):
    disp = Display(db_name)
    disp.draw_GUI()
    plt.tight_layout()
    disp.root.protocol("WM_DELETE_WINDOW", disp.quit)
    Tk.mainloop()

if __name__ == '__main__':
    disp_plot(db_name)
