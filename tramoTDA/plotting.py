import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams, PersistenceImager
from PIL import Image

def plot_trajectories(data, title, filename):
    """Plot multiple trajectories with a legend for each."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Loop through each trajectory and assign a label
    for index, trajectory in enumerate(data):
        label = f'Trajectory {index + 1}'  # Creating a label for each trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', label=label)

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()  # This will display the legend using the labels defined
    plt.savefig(filename)
    plt.close()

def plot_persistence_diagrams(diagrams, filename):
    """Plot multiple persistence diagrams with a legend for each, for an unspecified number of diagrams."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate a color map that has as many colors as diagrams
    colors = plt.cm.jet(np.linspace(0, 1, len(diagrams)))  # Uses a color map, can be changed to any other

    # Loop through each diagram and assign colors and labels dynamically
    for index, (diagram, color) in enumerate(zip(diagrams, colors)):
        label = f'Trajectory {index + 1}'  # Generating labels dynamically
        ax.scatter(diagram[:, 0], diagram[:, 1], color=color, label=label)  # Plot each diagram with a unique color

    ax.legend()  # This will display the legend using the labels defined
    ax.set_title("Persistence Diagrams")
    ax.set_xlabel('Birth')  # Typically the x-axis represents the birth time
    ax.set_ylabel('Death')  # Typically the y-axis represents the death time
    plt.savefig(filename)
    plt.close()

def plot_lifetime_diagrams(lifetimes, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lifetime in enumerate(lifetimes):
        ax.scatter(np.arange(len(lifetime)), lifetime, s=5, label=f'Trajectory {i+1}')
    ax.set_title("3. Lifetime Diagram")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Lifetime")
    ax.legend()
    plt.savefig(filename)
    plt.close()

def plot_persistence_images(diagrams_h1, filename):
    pimgr = PersistenceImager(pixel_size=0.1)
    pimgr.fit(diagrams_h1)
    imgs = pimgr.transform(diagrams_h1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, img in enumerate(imgs):
        axs[i].imshow(img, cmap='viridis', origin='lower')
        axs[i].set_title(f'Persistence Image {i+1}')
    plt.savefig(filename)
    plt.close()

def plot_classification(X, y, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    ax.set_title("6. Classification")
    plt.savefig(filename)
    plt.close()

def plot_evaluation_and_refinement(report, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.set_title("7. Evaluation and Refinement")
    ax.axis('off')
    plt.savefig(filename)
    plt.close()

def create_flowchart():
    image_paths = [
        ('1_Load_Trajectory_Data.png', 
        '1. Load Trajectory Data: Start by loading trajectory data, which could include spatial coordinates and timestamps from sources like GPS or AIS systems.'),
        ('2_Persistence_Diagrams.png', 
        '2. Generate Persistence Diagrams: Transform the trajectory data into persistence diagrams, capturing topological features that persist across various scales of the data. This step involves mathematical computations to identify significant structures within the data.'),
        ('3_Lifetime_Diagram.png', 
        '3. Generate Lifetime Diagrams: Calculate and plot the lifetime of each feature in the persistence diagrams to understand their persistence and significance over different scales.'),
        ('4_Persistence_Images.png', 
        '4. Generate Persistence Images: Visualize the persistence diagrams as images, providing a different perspective on the topological features.'),
        ('5_Calculate_Barycenter.png', 
        '5. Calculate Barycenter: Compute the barycenter of the persistence diagrams to find a representative summary of the data sets. This step helps in reducing complexity and summarizing the topological features.'),
        ('6_Classification_Logistic_Regression.png', 
        '6. Classification (Logistic Regression): Apply Logistic Regression to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('6_Classification_Support_Vector_Machine.png', 
        '6. Classification (Support Vector Machine): Apply Support Vector Machine (SVM) to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('6_Classification_Random_Forest.png', 
        '6. Classification (Random Forest): Apply Random Forest to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('6_Classification_Neural_Network.png', 
        '6. Classification (Neural Network): Apply Neural Network to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('7_Evaluation_and_Refinement_Logistic_Regression.png', 
        '7. Evaluation and Refinement (Logistic Regression): Assess the performance of the Logistic Regression model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.'),
        ('7_Evaluation_and_Refinement_Support_Vector_Machine.png', 
        '7. Evaluation and Refinement (Support Vector Machine): Assess the performance of the SVM model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.'),
        ('7_Evaluation_and_Refinement_Random_Forest.png', 
        '7. Evaluation and Refinement (Random Forest): Assess the performance of the Random Forest model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.'),
        ('7_Evaluation_and_Refinement_Neural_Network.png', 
        '7. Evaluation and Refinement (Neural Network): Assess the performance of the Neural Network model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.')
    ]


    fig, axs = plt.subplots(len(image_paths), 1, figsize=(10, 45))

    for i, (img_path, description) in enumerate(image_paths):
        axs[i].imshow(Image.new('RGB', (10, 10), color='white'))
        axs[i].axis('off')
        axs[i].text(0.5, 0.5, description, horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, fontsize=12, wrap=True)

    plt.tight_layout()
    plt.savefig('Flowchart_with_Descriptions.png')
    plt.show()