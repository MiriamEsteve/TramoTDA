import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from persim import plot_diagrams, PersistenceImager
import ripser
import gudhi
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary
from gudhi.persistence_graphical_tools import plot_persistence_diagram
from scipy.stats import norm
from PIL import Image

class TrajectoryAnalysis:
    def __init__(self, num_points=100, num_trajectories=3):
        self.num_points = num_points
        self.num_trajectories = num_trajectories
        self.datas = self.simulate_trajectory_data()

    def simulate_trajectory_data(self):
        np.random.seed(42)
        trajectories = []
        for _ in range(self.num_trajectories):
            x = np.cumsum(norm.rvs(size=self.num_points))
            y = np.cumsum(norm.rvs(size=self.num_points))
            trajectories.append(np.vstack((x, y)).T)
        return trajectories

    def plot_trajectory_data(self, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        for data, color, label in zip(self.datas, ['blue', 'green', 'red'], ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']):
            ax.plot(data[:, 0], data[:, 1], c=color, marker='o', linestyle='-', markersize=5, label=label)
        ax.set_title("1. Load Trajectory Data")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        ax.axis('equal')
        plt.savefig(filename)
        plt.close()

    def generate_persistence_diagrams(self):
        VR = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
        return [VR.fit_transform(data[None, :, :]) for data in self.datas]

    def plot_persistence_diagrams(self, diagrams, filename):
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

    def calculate_lifetime_diagrams(self, diagrams):
        return [diagram[0][:, 1] - diagram[0][:, 0] for diagram in diagrams]

    def plot_lifetime_diagrams(self, lifetimes, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, lifetime in enumerate(lifetimes):
            ax.scatter(np.arange(len(lifetime)), lifetime, s=5, label=f'Trajectory {i+1}')
        ax.set_title("3. Lifetime Diagram")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Lifetime")
        ax.legend()
        plt.savefig(filename)
        plt.close()

    def plot_persistence_images(self, diagrams_h1, filename):
        pimgr = PersistenceImager(pixel_size=0.1)
        pimgr.fit(diagrams_h1)
        imgs = pimgr.transform(diagrams_h1)
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for i, img in enumerate(imgs):
            axs[i].imshow(img, cmap='viridis', origin='lower')
            axs[i].set_title(f'Persistence Image {i+1}')
        plt.savefig(filename)
        plt.close()

    def proj_on_diag(self, x):
        return ((x[1] + x[0]) / 2, (x[1] + x[0]) / 2)

    def plot_bary(self, b, diags, G, ax):
        for i in range(len(diags)):
            indices = G[i]
            n_i = len(diags[i])

            for (y_j, x_i_j) in indices:
                y = b[y_j]
                if y[0] != y[1]:
                    if x_i_j >= 0:  # not mapped with the diag
                        x = diags[i][x_i_j]
                    else:  # y_j is matched to the diagonal
                        x = self.proj_on_diag(y)
                    ax.plot([y[0], x[0]], [y[1], x[1]], c='black',
                            linestyle="dashed")

        ax.scatter(b[:,0], b[:,1], color='purple', marker='d', label="barycenter (estim)")
        ax.legend()

    def compute_gudhi_barycenter(self, diags, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['blue', 'green', 'red']
        labels = ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']
        for diagram, color, label in zip(diags, colors, labels):
            plot_diagrams(diagram, ax=ax)
        handles = [plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=10, label=label) for color, label in zip(colors, labels)]
        ax.legend(handles=handles)

        b, log = bary(diags, 
             init=0,
             verbose=True)
        G = log["groupings"]
        self.plot_bary(b, diags, G, ax=ax)
        plt.savefig(filename)
        plt.close()

    def perform_classification(self):
        X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
        clf = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        report = classification_report(y_test, predictions)
        return X, y, report

    def plot_classification(self, X, y, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
        ax.set_title("6. Classification")
        plt.savefig(filename)
        plt.close()

    def plot_evaluation_and_refinement(self, report, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax.set_title("7. Evaluation and Refinement")
        ax.axis('off')
        plt.savefig(filename)
        plt.close()

    def create_flowchart(self):
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

    def run_analysis(self):
        self.plot_trajectory_data('1_Load_Trajectory_Data.png')
        
        diagrams = self.generate_persistence_diagrams()
        self.plot_persistence_diagrams(diagrams, '2_Persistence_Diagrams.png')
        
        lifetimes = self.calculate_lifetime_diagrams(diagrams)
        self.plot_lifetime_diagrams(lifetimes, '3_Lifetime_Diagram.png')
        
        rips = ripser.Rips(maxdim=1, coeff=2)
        diagrams_h1 = [rips.fit_transform(data)[1] for data in self.datas]
        self.plot_persistence_images(diagrams_h1, '4_Persistence_Images.png')

        self.compute_gudhi_barycenter(diagrams_h1, '5_Calculate_Barycenter.png')

        X, y, report = self.perform_classification()
        self.plot_classification(X, y, '6_Classification.png')
        self.plot_evaluation_and_refinement(report, '7_Evaluation_and_Refinement.png')

        self.create_flowchart()

if __name__ == "__main__":
    analysis = TrajectoryAnalysis()
    analysis.run_analysis()
