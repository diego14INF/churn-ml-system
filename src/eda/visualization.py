import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_eda_grid(df, features, target=None, cols=3):
    """
    Plot multiple features in a grid using appropriate visualization
    depending on feature type.

    Parameters
    ----------
    df : pandas.DataFrame
    features : list
        List of feature names to plot.
    target : str, optional
        Target variable for bivariate analysis.
    cols : int
        Number of columns in the grid.
    """

    # Filter out ID-like columns
    features = [
        f for f in features
        if df[f].nunique() < df.shape[0]
    ]

    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        # Numerical features
        if pd.api.types.is_numeric_dtype(df[feature]):
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f"{feature}")

        # Categorical features
        else:
            order = df[feature].value_counts().index
            sns.countplot(x=feature, data=df, order=order, ax=ax)
            ax.set_title(f"{feature}")
            ax.tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_eda_grid_vs_target(df, features, target, cols=3):
    features = [
        f for f in features
        if f != target and df[f].nunique() < df.shape[0]
    ]

    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        if pd.api.types.is_numeric_dtype(df[feature]):
            sns.boxplot(x=target, y=feature, data=df, ax=ax)
            ax.set_title(f"{feature} vs {target}")
        else:
            order = df[feature].value_counts().index
            sns.countplot(x=feature, hue=target, data=df, order=order, ax=ax)
            ax.set_title(f"{feature} vs {target}")
            ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


