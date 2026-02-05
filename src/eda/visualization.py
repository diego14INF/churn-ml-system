import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
# Set a friendly default renderer for VSCode notebooks; the caller can still override via the `renderer` argument
pio.renderers.default = 'vscode'



def plot_eda_grid_plotly(df, features, target=None, cols=3, max_categories=10, renderer=None):
    """
    Plot EDA grid fully in Plotly.
    - Numeric → Histogram (univariate) / Boxplot (vs target)
    - Categorical ≤ max_categories → Bar plot
    - Categorical > max_categories → Boxplot (to avoid label overlap)

    Parameters
    ----------
    renderer : str or None
        Optional plotly renderer to use when displaying the figure (e.g. 'vscode', 'notebook', 'browser').
    """

    # Remove ID-like columns
    features = [
        f for f in features
        if f != target and df[f].nunique() < df.shape[0]
    ]

    rows = math.ceil(len(features) / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=features
    )

    for i, feature in enumerate(features):
        row = i // cols + 1
        col = i % cols + 1

        # =========================
        # NUMERICAL VARIABLES
        # =========================
        if pd.api.types.is_numeric_dtype(df[feature]):

            if target is None:
                fig.add_trace(
                    go.Histogram(
                        x=df[feature],
                        nbinsx=30,
                        name=feature,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                for cls in df[target].unique():
                    fig.add_trace(
                        go.Box(
                            y=df[df[target] == cls][feature],
                            name=str(cls),
                            boxpoints='outliers',
                            showlegend=(i == 0)
                        ),
                        row=row, col=col
                    )

        # =========================
        # CATEGORICAL VARIABLES
        # =========================
        else:
            n_categories = df[feature].nunique()

            if target is None:
                vc = df[feature].value_counts()

                if n_categories <= max_categories:
                    fig.add_trace(
                        go.Bar(
                            x=vc.index.astype(str),
                            y=vc.values,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                else:
                    fig.add_trace(
                        go.Box(
                            y=vc.values,
                            boxpoints='all',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

            else:
                if n_categories <= max_categories:
                    for cls in df[target].unique():
                        vc = (
                            df[df[target] == cls][feature]
                            .value_counts()
                        )
                        fig.add_trace(
                            go.Bar(
                                x=vc.index.astype(str),
                                y=vc.values,
                                name=str(cls),
                                showlegend=(i == 0)
                            ),
                            row=row, col=col
                        )
                else:
                    for cls in df[target].unique():
                        fig.add_trace(
                            go.Box(
                                y=df[df[target] == cls][feature],
                                name=str(cls),
                                boxpoints='outliers',
                                showlegend=(i == 0)
                            ),
                            row=row, col=col
                        )

    fig.update_layout(
        height=250 * rows,
        width=250 * cols,
        title_text="EDA Grid (Plotly)",
        showlegend=True
    )

    # Use explicit renderer if provided (helps VSCode/Jupyter display correctly)
    if renderer is not None:
        fig.show(renderer=renderer)
    else:
        fig.show()




