import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
#from pytorch_grad_cam import EigenCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image
from einops import rearrange

matplotlib.use('Agg') # Workaround on qt 'xcb' loading problem.

def visualize_trisurf(x, y, z):
    fig = plt.figure()
    landscape = fig.add_subplot(projection='3d')
    landscape.plot_trisurf(x, y, z, alpha=0.8, cmap='viridis')
    landscape.set_title('Loss Landscape')
    landscape.set_xlabel('ε_1')
    landscape.set_ylabel('ε_2')
    landscape.set_zlabel('Loss')

    return landscape


def visualize_histogram(hist, bins):
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
    ax.set_xlim([bins.min(), bins.max()])
    ax.set_ylim([0, hist.max()])
    histogram = figure_to_array(fig)
    plt.close()

    return histogram


def visualize_plot(x, y, x_label, y_label, title):    
    fig, ax = plt.subplots()
    ax.plot(x,y)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    plot = figure_to_array(fig)
    plt.close()
    
    return plot


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def visualize_3d_q_values(num_actions, time_steps, q_values):
    x = np.repeat(np.arange(num_actions), time_steps)
    y = np.tile(np.arange(1, time_steps + 1), num_actions)
    z = q_values.flatten()
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Customize the plot
    ax.set_xlabel('Action')
    ax.set_ylabel('Time Steps')
    ax.set_zlabel('Q-Value')
    ax.set_title('3D Q-Value Plot')

    # Create lines connecting the dots based on time steps
    lines = []
    for i in range(num_actions):
        indices = np.arange(i * time_steps, (i + 1) * time_steps)
        line = np.column_stack([x[indices], y[indices], z[indices]])
        lines.append(line)

    # Create a Line3DCollection from the lines
    line_collection = Line3DCollection(lines, cmap='viridis', linewidths=0.2)

    # Add the Line3DCollection to the specific position along the 'z' axis
    ax.add_collection3d(line_collection, zs=0, zdir='z')
    
    return ax

class Eigen_CAM:
    '''
    Warning!
    Before running the code, please check two things.
    1. Change the code of 'grad-cam/base_cam.py'
       In forward(), you have to change 'outputs -> outputs[0]' like below
    
        if targets is None:
            target_categories = np.argmax(outputs[0].cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]
                
    2. Run with 'compile=False'
    '''
    def __init__(self, model, target_layers):
        self.cam = EigenCAM(model, [target_layers], use_cuda=False)
        
    def get_cam_image(self, x):
        # x -> (1, f, c, h, w)
        # grayscale_cam -> (84, 84)
        cam_input = rearrange(x[0], '1 f c h w -> 1 1 f c h w')
        grayscale_cam = self.cam(cam_input).squeeze(0)
        cam_input = rearrange(cam_input, '1 1 f c h w -> (1 1 f) c h w')
        cam_input = cam_input[-1][0].cpu().numpy()
        
        # show_cam_on_image must be 3 channels
        cam_input = np.stack((cam_input,)*3, axis=-1)
        
        # input data type must be float32
        cam_image = show_cam_on_image(np.float32(cam_input), grayscale_cam, use_rgb=True)
        cam_image = rearrange(cam_image, 'h w c -> c h w')
        
        return cam_image