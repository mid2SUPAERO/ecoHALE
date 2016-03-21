from __future__ import division
import tkFont
import Tkinter as Tk
from time import time
import sys

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
    NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy
import sqlitedict

import aluminum

#####################
# User-set parameters
#####################

if sys.argv[1] == 'as':
    filename = 'aerostruct'
elif sys.argv[1] == 'a':
    filename = 'weissinger'
elif sys.argv[1] == 's':
    filename = 'spatialbeam'
else:
    filename = sys.argv[1]

db_name = filename + '.db'

# TODO: ad theline
print aluminum.stress

def _get_lengths(self, A, B, axis):
    return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

class Display(object):
    def __init__(self, db_name):

        self.s = time()
        self.root = Tk.Tk()
        self.root.wm_title("Viewer")

        self.f = plt.figure(dpi=100, figsize=(12, 6), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.options_frame = Tk.Frame(self.root)
        self.options_frame.pack()

        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.ax = plt.subplot2grid((4,8), (0,0), rowspan=4, colspan=4, projection='3d')

        self.num_iters = 0
        self.db_name = db_name
        self.show_wing = True
        self.show_tube = True
        self.curr_pos = 0

        self.load_db()

        if self.show_wing and not self.show_tube:
            self.ax2 = plt.subplot2grid((4,8), (0,4), rowspan=2, colspan=4)
            self.ax3 = plt.subplot2grid((4,8), (2,4), rowspan=2, colspan=4)
        if self.show_tube and not self.show_wing:
            self.ax4 = plt.subplot2grid((4,8), (0,4), rowspan=2, colspan=4)
            self.ax5 = plt.subplot2grid((4,8), (2,4), rowspan=2, colspan=4)
        if self.show_wing and self.show_tube:
            self.ax2 = plt.subplot2grid((4,8), (0,4), colspan=4)
            self.ax3 = plt.subplot2grid((4,8), (1,4), colspan=4)
            self.ax4 = plt.subplot2grid((4,8), (2,4), colspan=4)
            self.ax5 = plt.subplot2grid((4,8), (3,4), colspan=4)

    def load_db(self):
        self.db = sqlitedict.SqliteDict(self.db_name, 'openmdao')
        self.twist = []
        self.mesh = []
        self.def_mesh = []
        self.r = []
        self.t = []
        sec_forces = []
        normals = []
        cos_dih = []
        self.lift = []
        self.vonmises = []

        for case_name, case_data in self.db.iteritems():
            if "metadata" in case_name or "derivs" in case_name:
                continue # don't plot these cases

            self.mesh.append(case_data['Unknowns']['mesh'])

            try:
                self.r.append(case_data['Unknowns']['r'])
                self.t.append(case_data['Unknowns']['t'])
                self.vonmises.append(
                    numpy.max(case_data['Unknowns']['vonmises'], axis=1))
                self.show_tube = True
            except:
                self.show_tube = False
                pass
            try:
                self.def_mesh.append(case_data['Unknowns']['def_mesh'])
                self.twist.append(case_data['Unknowns']['twist'])
                normals.append(case_data['Unknowns']['normals'])
                cos_dih.append(case_data['Unknowns']['cos_dih'])
                sec_forces.append(case_data['Unknowns']['sec_forces'])
                self.show_wing = True
            except:
                self.show_wing = False
                pass

        self.num_iters = numpy.max([len(self.mesh) - 1, 1])

        if self.show_wing:
            for i in range(self.num_iters + 1):
                L = sec_forces[i][:, 2] / normals[i][:, 2]
                self.lift.append(L.T * cos_dih[i])

            # recenter def_mesh points for better viewing
            for i in range(self.num_iters + 1):
                center = numpy.mean(numpy.mean(self.mesh[i], axis=0), axis=0)
                self.def_mesh[i] = self.def_mesh[i] - center

        # recenter mesh points for better viewing
        for i in range(self.num_iters + 1):
            center = numpy.mean(numpy.mean(self.mesh[i], axis=0), axis=0)
            self.mesh[i] = self.mesh[i] - center

        if self.show_wing:
            self.min_twist, self.max_twist = numpy.min(self.twist), numpy.max(self.twist)
            diff = (self.max_twist - self.min_twist) * 0.05
            self.min_twist -= diff
            self.max_twist += diff
            self.min_l, self.max_l = numpy.min(self.lift), numpy.max(self.lift)
            diff = (self.max_l - self.min_l) * 0.05
            self.min_l -= diff
            self.max_l += diff
        if self.show_tube:
            self.min_t, self.max_t = numpy.min(self.t), numpy.max(self.t)
            diff = (self.max_t - self.min_t) * 0.05
            self.min_t -= diff
            self.max_t += diff
            self.min_vm, self.max_vm = numpy.min(self.vonmises), numpy.max(self.vonmises)
            diff = (self.max_vm - self.min_vm) * 0.05
            self.min_vm -= diff
            self.max_vm += diff

    def plot_sides(self):
        m_vals = self.mesh[self.curr_pos]
        span = m_vals[0, :, 1] / m_vals[0, -1, 1]
        span_diff = (m_vals[0, :-1, 1] + m_vals[0, 1:, 1])/2 / m_vals[0, -1, 1]

        if self.show_wing:
            self.ax2.cla()
            self.ax3.cla()
            t_vals = self.twist[self.curr_pos]
            l_vals = self.lift[self.curr_pos]

            self.ax2.plot(span, t_vals, lw=2, c='b')
            self.ax2.locator_params(axis='y',nbins=4)
            self.ax2.locator_params(axis='x',nbins=3)
            self.ax2.set_ylim([self.min_twist, self.max_twist])
            self.ax2.set_ylabel('twist', rotation="horizontal", ha="right")

            self.ax3.plot(span_diff, l_vals, lw=2, c='b')
            self.ax3.locator_params(axis='y',nbins=4)
            self.ax3.locator_params(axis='x',nbins=3)
            self.ax3.set_ylim([self.min_l, self.max_l])
            self.ax3.set_ylabel('lift', rotation="horizontal", ha="right")

        if self.show_tube:
            self.ax4.cla()
            self.ax5.cla()
            thick_vals = self.t[self.curr_pos]
            vm_vals = self.vonmises[self.curr_pos]

            self.ax4.plot(span_diff, thick_vals, lw=2, c='b')
            self.ax4.locator_params(axis='y',nbins=4)
            self.ax4.locator_params(axis='x',nbins=3)
            self.ax4.set_ylim([self.min_t, self.max_t])
            self.ax4.set_ylabel('thickness', rotation="horizontal", ha="right")

            self.ax5.plot(span_diff, vm_vals, lw=2, c='b')
            self.ax5.locator_params(axis='y',nbins=4)
            self.ax5.locator_params(axis='x',nbins=3)
            self.ax5.set_ylim([self.min_vm, self.max_vm])
            self.ax5.set_ylabel('von mises', rotation="horizontal", ha="right")

    def plot_wing(self):
        self.ax.cla()
        az = self.ax.azim
        el = self.ax.elev
        dist = self.ax.dist
        mesh0 = self.mesh[self.curr_pos].copy()

        self.ax.set_axis_off()

        if self.show_wing:
            def_mesh0 = self.def_mesh[self.curr_pos]
            x = mesh0[:, :, 0]
            y = mesh0[:, :, 1]
            z = mesh0[:, :, 2]

            if self.show_def_mesh.get():
                x_def = def_mesh0[:, :, 0]
                y_def = def_mesh0[:, :, 1]
                z_def = def_mesh0[:, :, 2]

                self.c2.grid(row=0, column=3, padx=5, sticky=Tk.W)
                if self.ex_def.get():
                    z_def = (z_def - z) * 10 + z_def
                    def_mesh0 = (def_mesh0 - mesh0) * 30 + def_mesh0
                else:
                    def_mesh0 = (def_mesh0 - mesh0) * 2 + def_mesh0
                self.ax.plot_wireframe(x_def, y_def, z_def, rstride=1, cstride=1, color='b')
                self.ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k', alpha=.5)
            else:
                self.ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k')
                self.c2.grid_forget()

        if self.show_tube:
            r0 = self.r[self.curr_pos]
            t0 = self.t[self.curr_pos]
            colors = t0
            colors = colors / numpy.max(colors)
            num_circ = 12
            fem_origin = 0.35
            n = mesh0.shape[1]
            p = numpy.linspace(0, 2*numpy.pi, num_circ)
            if self.show_wing:
                if self.show_def_mesh.get():
                    mesh0[:, :, 2] = def_mesh0[:, :, 2]
            for i, thick in enumerate(t0):
                r = numpy.array((r0[i], r0[i]))
                R, P = numpy.meshgrid(r, p)
                X, Z = R*numpy.cos(P), R*numpy.sin(P)
                chords = mesh0[1, :, 0] - mesh0[0, :, 0]
                comp = fem_origin * chords + mesh0[0, :, 0]
                X[:, 0] += comp[i]
                X[:, 1] += comp[i+1]
                Z[:, 0] += fem_origin * mesh0[1, i, 2]
                Z[:, 1] += fem_origin * mesh0[1, i+1, 2]
                Y = numpy.empty(X.shape)
                Y[:] = numpy.linspace(mesh0[0, i, 1], mesh0[0, i+1, 1], 2)
                col = numpy.zeros(X.shape)
                col[:] = colors[i]
                self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=cm.YlOrRd(col), linewidth=0)
        lim = numpy.max(numpy.max(mesh0)) / 3
        self.ax.auto_scale_xyz([-lim, lim], [-lim, lim], [-lim, lim])
        self.ax.set_title("Iteration: {}".format(self.curr_pos))

        self.ax.view_init(elev=el, azim=az) #Reproduce view
        self.ax.dist = dist

    def update_graphs(self, e=None):
        if e != None:
            self.curr_pos = int(e)
            self.curr_pos = self.curr_pos % (self.num_iters + 1)

        self.plot_wing()
        self.plot_sides()
        self.canvas.show()

    def quit(self):
        """
        Destroy GUI window cleanly if quit button pressed.
        """
        self.root.quit()
        self.root.destroy()

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

        # scale to choose iteration to view
        self.w = Tk.Scale(
            self.options_frame,
            from_=0, to=self.num_iters,
            orient=Tk.HORIZONTAL,
            resolution=1,
            font=font,
            command=self.update_graphs,
            length=200)
        self.w.set(0)
        self.w.grid(row=0, column=1, padx=5, sticky=Tk.W)

        if self.show_wing:
            # checkbox to show deformed mesh
            self.show_def_mesh = Tk.IntVar()
            c1 = Tk.Checkbutton(
                self.options_frame,
                text="Show deformed mesh",
                variable=self.show_def_mesh,
                command=self.update_graphs,
                font=font)
            c1.grid(row=0, column=2, padx=5, sticky=Tk.W)

            # checkbox to exaggerated deformed mesh
            self.ex_def = Tk.IntVar()
            self.c2 = Tk.Checkbutton(
                self.options_frame,
                text="Exaggerate deformations",
                variable=self.ex_def,
                command=self.update_graphs,
                font=font)
            self.c2.grid(row=0, column=3, padx=5, sticky=Tk.W)


def disp_plot(db_name):
    disp = Display(db_name)
    disp.draw_GUI()
    plt.tight_layout()
    disp.root.protocol("WM_DELETE_WINDOW", disp.quit)
    Tk.mainloop()

if __name__ == '__main__':
    disp_plot(db_name)
