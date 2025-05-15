import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_balls_trail(csv_file):
    """
    this function plots the 3d sign using matplot lib.
    it also finds the mean and std of the distance between the balls. 
    This is because, realistically the balls should be at a const distance from each other
    """
    df = pd.read_csv(csv_file)
    
    if df.shape[1] != 6:
        print("Error: The CSV file must have exactly 6 columns (x, y, z for ball1 and x, y, z for ball2).")
        return

    #columns are: [x1, y1, z1, x2, y2, z2]
    ball1_x = df.iloc[:, 0]
    ball1_y = df.iloc[:, 1]
    ball1_z = df.iloc[:, 2]
    
    ball2_x = df.iloc[:, 3]
    ball2_y = df.iloc[:, 4]
    ball2_z = df.iloc[:, 5]

    distances = np.sqrt((ball2_x - ball1_x)**2 +
                        (ball2_y - ball1_y)**2 +
                        (ball2_z - ball1_z)**2)

    mean_distance = distances.mean()
    std_distance = distances.std()
    print(f"Mean distance between the balls: {mean_distance:.2f}")
    print(f"Standard deviation of distances: {std_distance:.2f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(ball1_x, ball1_y, ball1_z, color='red', label='Tip trajectory')
    ax.plot(ball2_x, ball2_y, ball2_z, color='green', label='Tail trajectory')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    plot_balls_trail(csv_file_path)
