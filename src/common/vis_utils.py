import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List
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


def visualize_multi_images(image_list, model_types, rows, cols, mode, save_path):
    """
    Visualizes multiple images with labels in a grid format and saves the figure.

    Parameters:
    image_paths (list): List of paths to images to be displayed.
    model_types (dict): Dictionary of categories and their corresponding models.
    save_path (str): Path where the figure will be saved.
    """    
    if len(image_list) != sum(len(models) for models in model_types.values()):
        raise ValueError("The number of images does not match the sum of model types.")

    if rows * cols != len(image_list):
        raise ValueError("The number of images and the grid size do not match.")
    
    _model_types = {key: [model.replace('_', '-') for model in models] for key, models in model_types.items()}
    

    # Dynamically generate ordered_models list
    ordered_models = [model for sublist in _model_types.values() for model in sublist]

    # Calculate the start positions for each model type
    excluded_keys = ['Input', 'Blank']
    group_starts = {}
    counter = 0
    for category in _model_types.keys():
        if category not in excluded_keys:
            group_starts[f'{category}']=counter
        counter += len(_model_types[category])
      
    if mode == 0:
        # Create the figure
        fig = plt.figure(figsize=(cols * 2.6, rows * 3.1))

        # Creating subplots with adjusted positions
        for i, model_name in enumerate(ordered_models):
            row, col = divmod(i, cols)
            if col == 0:
                ax_left = 0.005
            ax_bottom = 1 - (row + 1) / rows  + 0.07 # Decrease for space between rows
            ax_width = 1 / cols - 0.005  # Slight reduction to prevent overlap
            ax_height = 1 / rows - 0.08  # Adjust for space between rows
            
            # Increase space at the start of a new model type
            if i in group_starts.values():
                ax_left += 0.035  # Adjust this value to increase space
            # import pdb; pdb.set_trace()
            
            ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
            print(f'\n {i+1}.')
            print(f'left: {ax_left}')
            print(f'bottom: {ax_bottom}')
            print(f'width: {ax_width}')
            print(f'height: {ax_height}')
            ax_left += ax_width

            # Display image
            ax.imshow(image_list[i])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if model_name.startswith('Blank'):
                ax.axis('off')
            else:
                ax.set_xlabel(model_name, fontfamily='serif', fontsize=18, labelpad=5)
                
            ax.set_xticks([])
            ax.set_yticks([])

            # Set y-axis labels for specified groups
            if i in group_starts.values():
                # import pdb; pdb.set_trace()
                label = list(group_starts.keys())[list(group_starts.values()).index(i)]
                ax.set_ylabel(label, fontfamily='serif', fontsize=20, labelpad=10, rotation=90, ha='center', va='center')
        
    elif mode == 1:
        # Create the figure
        if cols ==7:
            fig = plt.figure(figsize=(cols * 2.75, rows * 3.3))
        else:
            fig = plt.figure(figsize=(cols * 2.3, rows * 3))

        # Creating subplots with adjusted positions
        for i, model_name in enumerate(ordered_models):
            row, col = divmod(i, cols)
            if cols==7:
                if col == 0:
                    ax_left = 0.005
            else:
                if col == 0:
                    ax_left = 0.015
            ax_bottom = 1 - (row + 1) / rows  + 0.07 # Decrease for space between rows
            ax_width = 1 / cols - 0.01  # Slight reduction to prevent overlap
            ax_height = 1 / rows - 0.13  # Adjust for space between rows
            
            # Increase space at the start of a new model type
            if i in group_starts.values() and col>0:
                ax_left += 0.025  # Adjust this value to increase space
            # import pdb; pdb.set_trace()
            
            ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
            # print(f'\n {i+1}.')
            # print(f'left: {ax_left}')
            # print(f'bottom: {ax_bottom}')
            # print(f'width: {ax_width}')
            # print(f'height: {ax_height}')
            ax_left += ax_width

            # Display image
            ax.imshow(image_list[i])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if model_name.startswith('Blank'):
                ax.axis('off')
            # else:
            #     ax.set_xlabel(model_name, fontfamily='serif', fontsize=18, labelpad=5)
                
            ax.set_xticks([])
            ax.set_yticks([])
            # Display group label above the first image of each group
            # if i-1 in group_starts.values():
            #     group_label = list(group_starts.keys())[list(group_starts.values()).index(i-1)]
            #     # Adjust text position based on the subplot position
            #     if cols ==7:
            #         plt.text(ax_left-0.067, ax_bottom + ax_height + 0.03,  # Adjust these values as needed
            #                 group_label, fontfamily='serif', fontsize=20, ha='center', va='center', transform=fig.transFigure)
            #     else:
            #         plt.text(ax_left-0.079, ax_bottom + ax_height + 0.031,  # Adjust these values as needed
            #                 group_label, fontfamily='NanumGothic', fontsize=20, ha='center', va='center', weight='semibold', transform=fig.transFigure)
                
    elif mode == 2:
        # Create the figure
        fig = plt.figure(figsize=(cols * 2.6, rows * 3))

        # Creating subplots with adjusted positions
        for i, model_name in enumerate(ordered_models):
            row, col = divmod(i, cols)
            if col == 0:
                ax_left = 0.005
            ax_bottom = 1 - (row + 1) / rows  + 0.07 # Decrease for space between rows
            ax_width = 1 / cols - 0.01  # Slight reduction to prevent overlap
            ax_height = 1 / rows - 0.09  # Adjust for space between rows
            
            # Increase space at the start of a new model type
            if i in group_starts.values() and i in [1, 8]:
                ax_left += 0.025  # Adjust this value to increase space
            elif i in group_starts.values() and i not in [1, 8]:
                ax_left += 0.025  # Adjust this value to increase space
            # import pdb; pdb.set_trace()
            
            ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
            print(f'\n {i+1}.')
            print(f'left: {ax_left}')
            print(f'bottom: {ax_bottom}')
            print(f'width: {ax_width}')
            print(f'height: {ax_height}')
            ax_left += ax_width

            # Display image
            ax.imshow(image_list[i])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if model_name.startswith('Blank'):
                ax.axis('off')
            else:
                ax.set_xlabel(model_name, fontfamily='serif', fontsize=18, labelpad=5)
                
            ax.set_xticks([])
            ax.set_yticks([])
            
    elif mode == 3:
        import matplotlib.patches as patches
        
        fig, axs = plt.subplots(rows, cols, figsize=(15, 3.6))

        # Flatten the axis array for easy indexing
        axs = axs.ravel()

        for i, (img, model) in enumerate(zip(image_list, ordered_models)):
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(model, fontfamily='serif', fontsize=20)  # Set padding for the title

            # # Adding extra space before the first image
            # if i == 0:
            #     axs[i].set_frame_on(False)
            #     axs[i].add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', lw=0))

            # # Adjusting spacing between images
            # if i == 1:
            #     plt.setp(axs[i], xmargin=0.05)


        plt.tight_layout()



    plt.savefig(save_path)
    plt.close()
    
    
class Eigen_CAM(EigenCAM):
    '''
        Warning!
        Before running the code, please make 'compile=false'
    '''
    def __init__(self, model, target_layers, use_cuda=False):
        super().__init__(model, target_layers, use_cuda)
        
    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
            
        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs[0].cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]
        
        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)
            
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        
        return self.aggregate_multi_layers(cam_per_layer)
        
    def get_eigencam_image(self, x):
        # x -> (1, f, c, h, w)
        # grayscale_cam -> (84, 84)
        cam_input = rearrange(x[0], '1 f c h w -> 1 1 f c h w')
        grayscale_cam = self(cam_input).squeeze(0)
        cam_input = rearrange(cam_input, '1 1 f c h w -> (1 1 f) c h w')
        cam_input = cam_input[-1][0].cpu().numpy()
        
        # show_cam_on_image must be 3 channels
        cam_input = np.stack((cam_input,)*3, axis=-1)
        
        # input data type must be float32
        cam_image = show_cam_on_image(np.float32(cam_input), grayscale_cam, use_rgb=True)
        cam_image = rearrange(cam_image, 'h w c -> c h w')
        
        return cam_image