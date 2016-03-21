import plotly.graph_objs as go
import matplotlib.pyplot as plt

def wire_mesh(mesh):
    le = mesh[0]
    te = mesh[1]

    n_points = len(le)

    lines = []
    line_marker = dict(color='#0066FF', width=2)
    for i in xrange(n_points):
        s = go.Scatter3d(x=[le[i,0],te[i,0]], y=[le[i,1],te[i,1]], z=[le[i,2],te[i,2]], mode='lines', line=line_marker)
        lines.append(s)

        if i < n_points-1:
            s = go.Scatter3d(x=[le[i,0],le[i+1,0]], y=[le[i,1],le[i+1,1]], z=[le[i,2],le[i+1,2]], mode='lines', line=line_marker)
            lines.append(s)
            s = go.Scatter3d(x=[te[i,0],te[i+1,0]], y=[te[i,1],te[i+1,1]], z=[te[i,2],te[i+1,2]], mode='lines', line=line_marker)
            lines.append(s)

    return lines


def build_layout():
    layout = go.Layout(
        title='Wireframe Plot',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectmode="data"

        ),
        showlegend=False,
    )

    return layout


def adjust_spines(ax = None, spines=['left', 'bottom'], off_spines=['top', 'right']):
    if ax == None:
        ax = plt.gca()

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 18))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    for spine in off_spines:
        ax.spines[spine].set_visible(False)
