from tramoTDA.analysis import TrajectoryAnalysis
from tramoTDA.plotting import (
    plot_trajectory_Brownian, plot_trajectory_Levy, plot_trajectory_Spiral, plot_trajectory_Circular,
    plot_trajectory_data, plot_persistence_diagrams, plot_lifetime_diagrams, 
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


def main_all():
    analysis = TrajectoryAnalysis()
    analysis.run_analysis()

def main():
    # Initialize the TrajectoryAnalysis class
    analysis = TrajectoryAnalysis()
    
    # Step 1: Plot Trajectory Data
    data_gen = analysis.SimulatedTrajectoryData()
    brownian_data = data_gen.brownian_motion()
    levy_data = data_gen.levy_flight()
    spiral_data = data_gen.spiral_trajectory()
    circular_data = data_gen.circular_trajectory()

    # Plotting example trajectories
    plot_trajectory_Brownian(brownian_data, '1_Load_Trajectory_Brownian_Motion.png')
    plot_trajectory_Levy(levy_data, '1_Load_Trajectory_Levy_Flight.png')
    plot_trajectory_Spiral(spiral_data, '1_Load_Trajectory_Spiral.png')
    plot_trajectory_Circular(circular_data, '1_Load_Trajectory_Circular.png')
    
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
        plot_classification(X, y, f'6_Classification_{name.replace(" ", "_")}.png')
        plot_evaluation_and_refinement(report, f'7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png')

    # Step 7: Create Flowchart
    create_flowchart()

if __name__ == "__main__":
    # Execute all the analysis
    # main_all()

    # Execute step by step
    main()
