from tramoTDA.analysis import TrajectoryAnalysis
from tramoTDA.data import SimulatedTrajectoryData
from tramoTDA.plotting import (
    plot_trajectories, plot_persistence_diagrams, plot_lifetime_diagrams, 
    plot_persistence_images, plot_classification, plot_evaluation_and_refinement, create_flowchart
)
from tramoTDA.utils import (
    generate_persistence_diagrams, calculate_lifetime_diagrams, 
    compute_gudhi_barycenter, perform_classification
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import ripser
import matplotlib.pyplot as plt

def main():
    # Step 1: Plot Trajectory Data
    data_gen = SimulatedTrajectoryData()
    brownian_data = data_gen.brownian_motion()
    levy_data = data_gen.levy_flight()
    spiral_data = data_gen.spiral_trajectory()
    circular_data = data_gen.circular_trajectory()

    # Plotting example trajectories
    plot_trajectories(brownian_data, 'Trajectory_Brownian_Motion', '1_Load_Trajectory_Brownian_Motion.png')
    plot_trajectories(levy_data, 'Trajectory_Levy_Flight', '1_Load_Trajectory_Levy_Flight.png')
    plot_trajectories(spiral_data, 'Trajectory_Spiral', '1_Load_Trajectory_Spiral.png')
    plot_trajectories(circular_data, 'Trajectory_Circular', '1_Load_Trajectory_Circular.png')
    
    # Initialize the TrajectoryAnalysis class
    analysis = TrajectoryAnalysis()
    analysis.datas = data_gen.simulate_trajectory_data()

    # Step 2: Generate Persistence Diagrams
    diagrams = generate_persistence_diagrams(analysis.datas)
    plot_persistence_diagrams(diagrams, '2_Persistence_Diagrams.png')
    
    # Step 3: Calculate and Plot Lifetime Diagrams
    lifetimes = calculate_lifetime_diagrams(diagrams)
    plot_lifetime_diagrams(lifetimes, '3_Lifetime_Diagram.png')
    
    # Step 4: Generate Persistence Images
    rips_instance = ripser.Rips(maxdim=1, coeff=2)
    diagrams_h1 = [rips_instance.fit_transform(data)[1] for data in analysis.datas]
    plot_persistence_images(diagrams_h1, '4_Persistence_Images.png')

    # Step 5: Compute and Plot Barycenter
    compute_gudhi_barycenter(diagrams_h1, '5_Calculate_Barycenter.png')

    # Step 6: Perform Classification with Different Models
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Neural Network': MLPClassifier(max_iter=2000)  # Increased max_iter to 2000
    }
    
    for name, clf in classifiers.items():
        X, y, report = perform_classification(clf)
        plot_classification(X, y, f'6_Classification_{name.replace(" ", "_")}.png', name)
        plot_evaluation_and_refinement(report, f'7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)

    # Step 7: Create Flowchart
    create_flowchart()

if __name__ == "__main__":
    # Execute all the analysis
    # main_all()

    # Execute step by step
    main()
