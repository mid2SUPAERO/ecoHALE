import plotly.graph_objs as go


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