import matplotlib.pyplot as plt
from persim import plot_diagrams, PersistenceImager
from PIL import Image

def plot_trajectory_data(datas, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    for data, color, label in zip(datas, ['blue', 'green', 'red'], ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']):
        ax.plot(data[:, 0], data[:, 1], c=color, marker='o', linestyle='-', markersize=5, label=label)
    ax.set_title("1. Load Trajectory Data")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.axis('equal')
    plt.savefig(filename)
    plt.close()

def plot_persistence_diagrams(diagrams, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    labels = ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']
    for diagram, color, label in zip(diagrams, colors, labels):
        plot_diagrams(diagram[0], ax=ax)
    handles = [plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=10, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=handles)
    ax.set_title("2. Generate Persistence Diagrams")
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
        ('1_Load_Trajectory_Data.png', '1. Load Trajectory Data: Start by loading trajectory data, which could include spatial coordinates and timestamps from sources like GPS or AIS systems.'),
        ('2_Persistence_Diagrams.png', '2. Generate Persistence Diagrams: Transform the trajectory data into persistence diagrams, capturing topological features that persist across various scales of the data. This step involves mathematical computations to identify significant structures within the data.'),
        ('3_Lifetime_Diagram.png', '3. Generate Lifetime Diagrams: Calculate and plot the lifetime of each feature in the persistence diagrams.'),
        ('4_Persistence_Images.png', '4. Generate Persistence Images: Visualize the persistence diagrams as images.'),
        ('5_Calculate_Barycenter.png', '5. Calculate Barycenter: Compute the barycenter of the persistence diagrams to find a representative summary of the data sets. This step helps in reducing complexity and summarizing the topological features.'),
        ('6_Classification.png', '6. Classification: Apply machine learning algorithms to classify the trajectories based on the features derived from the persistence diagrams and their barycenters. This might involve using classifiers like SVM, Random Forest, or neural networks, depending on the complexity and nature of the data.'),
        ('7_Evaluation_and_Refinement.png', '7. Evaluation and Refinement: Assess the performance of the classification models using metrics such as accuracy, precision, and recall. Refine the models based on the outcomes to improve classification results.')
    ]
    
    fig, axs = plt.subplots(len(image_paths), 1, figsize=(10, 45))
    for i, (img_path, description) in enumerate(image_paths):
        img = Image.open(img_path)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].text(0.5, -0.1, description, horizontalalignment='center', verticalalignment='top', transform=axs[i].transAxes, fontsize=12, wrap=True)
    
    plt.tight_layout()
    plt.savefig('Flowchart_with_Descriptions.png')
    plt.show()
