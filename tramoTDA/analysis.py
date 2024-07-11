import numpy as np
from .data import SimulatedTrajectoryData
from .plotting import (
    plot_trajectories, plot_persistence_diagrams, plot_lifetime_diagrams, 
    plot_persistence_images, plot_classification, plot_evaluation_and_refinement, create_flowchart
)
from .utils import (
    generate_persistence_diagrams, calculate_lifetime_diagrams, 
    compute_gudhi_barycenter, perform_classification
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import ripser

class TrajectoryAnalysis:
    def __init__(self, num_trajectories=10, num_steps=100, data=SimulatedTrajectoryData(10, 100).simulate_trajectory_data()):
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps
        self.datas = data

    def run_analysis(self):
        plot_trajectories(self.datas, 'Load_Trajectory_Data', '1_Load_Trajectory_Data.png')
        
        diagrams = generate_persistence_diagrams(self.datas)
        plot_persistence_diagrams(diagrams, '2_Persistence_Diagrams.png')
        
        lifetimes = calculate_lifetime_diagrams(diagrams)
        plot_lifetime_diagrams(lifetimes, '3_Lifetime_Diagram.png')
        
        rips = ripser.Rips(maxdim=1, coeff=2)
        diagrams_h1 = [rips.fit_transform(data)[1] for data in self.datas]
        plot_persistence_images(diagrams_h1, '4_Persistence_Images.png')

        compute_gudhi_barycenter(diagrams_h1, '5_Calculate_Barycenter.png')

        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'Support Vector Machine': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Neural Network': MLPClassifier()
        }
        
        for name, clf in classifiers.items():
            X, y, report = perform_classification(clf)
            plot_classification(X, y, f'6_Classification_{name.replace(" ", "_")}.png', name)
            plot_evaluation_and_refinement(report, f'7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)

        create_flowchart()

