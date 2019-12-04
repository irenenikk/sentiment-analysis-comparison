import matplotlib.pyplot as plt

def plot_heat_map(matrix):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()
