import wandb
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image


def pca_plot(x_0_gt, x_1_gt, x_1_pred, n_plot, step, lims=((-10, 10), (-10, 10))):
    fig,axes = plt.subplots(1, 3,figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    pca = PCA(n_components=2).fit(x_1_gt)
    
    x_0_gt_pca = pca.transform(x_0_gt[:n_plot])
    x_1_gt_pca = pca.transform(x_1_gt[:n_plot])
    x_1_pred_pca = pca.transform(x_1_pred[:n_plot])
    
    axes[0].scatter(x_0_gt_pca[:,0], x_0_gt_pca[:,1], c="g", edgecolor = 'black',
                    label = r'$x\sim P_0(x)$', s =30)
    axes[1].scatter(x_1_gt_pca[:,0], x_1_gt_pca[:,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s =30)
    axes[2].scatter(x_1_pred_pca[:,0], x_1_pred_pca[:,1], c="yellow", edgecolor = 'black',
                    label = r'$x\sim T(x)$', s =30)
    
    for i in range(3):
        axes[i].grid()
        axes[i].set_xlim(lims[0])
        axes[i].set_ylim(lims[1])
        axes[i].legend()
    
    fig.tight_layout(pad=0.5)
    wandb.log({f'Plot PCA samples' : [wandb.Image(fig2img(fig))]}, step=step)

    
def plot_mapping(independent_mapping, true_mapping, predicted_mapping, target_data, n_plot, step):
    s=30
    linewidth=0.2
    map_alpha=1
    data_alpha=1
    figsize=(5, 5)
    dpi=None
    data_color='red'
    mapped_data_color='blue'
    map_color='green'
    map_label=None
    data_label=None
    mapped_data_label=None
    
    dim = target_data.shape[-1]
    pca = PCA(n_components=2).fit(target_data)
    
    independent_mapping_pca = np.concatenate((        
        pca.transform(independent_mapping[:n_plot, :dim]),
        pca.transform(independent_mapping[:n_plot, dim:]),
        ), axis=-1)
 
    true_mapping_pca = np.concatenate((
        pca.transform(true_mapping[:n_plot, :dim]),
        pca.transform(true_mapping[:n_plot, dim:]),
    ), axis=-1)
    
    predicted_mapping_pca = np.concatenate((
        pca.transform(predicted_mapping[:n_plot, :dim]),
        pca.transform(predicted_mapping[:n_plot, dim:]),
    ), axis=-1)
    
    target_data_pca = pca.transform(target_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    titles = ["independent", "true", "predicted"]
    for i, mapping in enumerate([independent_mapping_pca, true_mapping_pca, predicted_mapping_pca]):
        inp = mapping[:, :2]
        out = mapping[:, 2:]

        lines = np.concatenate([inp, out], axis=-1).reshape(-1, 2, 2)
        lc = matplotlib.collections.LineCollection(
            lines, color=map_color, linewidths=linewidth, alpha=map_alpha, label=map_label)
        axes[i].add_collection(lc)

        axes[i].scatter(
            inp[:, 0], inp[:, 1], s=s, label=data_label,
            alpha=data_alpha, zorder=2, color=data_color)
        axes[i].scatter(
            out[:, 0], out[:, 1], s=s, label=mapped_data_label,
            alpha=data_alpha, zorder=2, color=mapped_data_color)

        axes[i].scatter(target_data_pca[:1000,0], target_data_pca[:1000,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s =10)
        axes[i].grid()
        axes[i].set_title(titles[i])
    
    wandb.log({f'Plot PCA plan samples' : [wandb.Image(fig2img(fig))]}, step=step)

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
    
def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def plot_2D(x_0_gt, x_1_gt, x_1_pred, n_plot, step, lims=((-2.25, 2.5), (-2.25, 2.5))):
    fig,axes = plt.subplots(1, 3,figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    
    x_0_gt_pca = x_0_gt[:n_plot]
    x_1_gt_pca = x_1_gt[:n_plot]
    x_1_pred_pca = x_1_pred[:n_plot]
    
    axes[0].scatter(x_0_gt[:,0], x_0_gt_pca[:,1], c="g", edgecolor = 'black',
                    label = r'$x\sim P_0(x)$', s = 16)
    axes[1].scatter(x_1_gt[:,0], x_1_gt_pca[:,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s = 16)
    axes[2].scatter(x_1_pred[:,0], x_1_pred_pca[:,1], c="yellow", edgecolor = 'black',
                    label = r'$x\sim T(x)$', s = 16)
    
    for i in range(3):
        axes[i].set_xlim(*lims[0])
        axes[i].set_ylim(*lims[1])
        axes[i].grid()
        axes[i].legend()
    
    fig.tight_layout(pad=0.5)
    wandb.log({f'Plot samples' : [wandb.Image(fig2img(fig))]}, step=step)
    
    
def plot_2D_trajectory(x_0, x_1, trajectory, step, lims=((-2.25, 2.5), (-2.25/1.5, 2.5/1.5)), x_scatter_alpha=1):
    fig,axes = plt.subplots(1, 1,figsize=(6,4),squeeze=True,sharex=True,sharey=True)
    
    axes.scatter(x_0[:, 0], x_0[:, 1], c="g", edgecolor = 'black',
                    label = r'$x\sim P_0(x)$', s = 16, alpha=x_scatter_alpha)
    axes.scatter(x_1[:, 0], x_1[:, 1], c="orange", edgecolor = 'black',
                    label = r'$x\sim \pi(\cdot|x)$', s = 16)
    
    axes.grid()
    axes.legend()
    
    axes.set_xlim(*lims[0])
    axes.set_ylim(*lims[1])
    
    for i in range(5):
        plt.plot(trajectory[i, :, 0], trajectory[i, :, 1], "-o", markeredgecolor="black", linewidth=4, markersize=4)
    
    fig.tight_layout(pad=0.5)
    wandb.log({f'Plot trajectory samples' : [wandb.Image(fig2img(fig))]}, step=step)

    
def plot_2D_mapping(independent_mapping, predicted_mapping,
                    target_data, n_plot, step, lims=((-2.25, 2.5), (-2.25/1.5, 2.5/1.5))):
    s=30
    linewidth=0.2
    map_alpha=1
    data_alpha=1
    figsize=(5, 5)
    dpi=None
    data_color='red'
    mapped_data_color='blue'
    map_color='green'
    map_label=None
    data_label=None
    mapped_data_label=None
    
    dim = target_data.shape[-1]
    
    independent_mapping_pca = np.concatenate((independent_mapping[:n_plot, :dim], independent_mapping[:n_plot, dim:]), axis=-1)
    predicted_mapping_pca = np.concatenate((predicted_mapping[:n_plot, :dim], predicted_mapping[:n_plot, dim:]), axis=-1)

    target_data_pca = target_data
    
    fig, axes = plt.subplots(1, 2, figsize=(8,4),squeeze=True,sharex=True,sharey=True)
    titles = ["independent", "predicted"]
    for i, mapping in enumerate([independent_mapping_pca, predicted_mapping_pca]):
        inp = mapping[:, :2]
        out = mapping[:, 2:]

        lines = np.concatenate([inp, out], axis=-1).reshape(-1, 2, 2)
        lc = matplotlib.collections.LineCollection(
            lines, color=map_color, linewidths=linewidth, alpha=map_alpha, label=map_label)
        axes[i].add_collection(lc)

        axes[i].scatter(
            inp[:, 0], inp[:, 1], s=s, label=data_label,
            alpha=data_alpha, zorder=2, color=data_color)
        axes[i].scatter(
            out[:, 0], out[:, 1], s=s, label=mapped_data_label,
            alpha=data_alpha, zorder=2, color=mapped_data_color)

        axes[i].scatter(target_data_pca[:1000,0], target_data_pca[:1000,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s = 16)
        axes[i].set_title(titles[i])
        
        axes[i].set_xlim(*lims[0])
        axes[i].set_ylim(*lims[1])
        axes[i].grid()
    
    wandb.log({f'Plot plan samples' : [wandb.Image(fig2img(fig))]}, step=step)

 