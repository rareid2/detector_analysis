import numpy as np
import matplotlib.pyplot as plt

def read_two_column_file(filename):
    column1 = []
    column2 = []

    with open(filename, 'r') as file:
        for line in file:
            # Split the line into two columns based on whitespace
            col1, col2 = map(float, line.split())
            column1.append(col1)
            column2.append(col2)
    dataset = np.array([column2, column1])

    return dataset


def plot_subplots(datasets, subplot_titles):
    """Create three subplots with five colored lines in each."""
    num_subplots = len(datasets)
    num_lines = len(datasets[0])
    res_deg = ["+40", "17--14", "8--7.5", "5.3--5.2", "4"]

    colors = ['#03045E', '#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8']  # You can customize the colors as needed.

    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 12), sharex = True, sharey=True)

    for i, ax in enumerate(axs):
        ax.set_title(subplot_titles[i])
        
        ax.set_ylabel("normalized intensity")

        for j, res in zip(range(num_lines), res_deg):
            x = datasets[i][j][0]
            y = datasets[i][j][1]

            ax.plot(x,y, color=colors[j], label=f'~ {res} deg')

        ax.legend(loc='upper right')
    ax.set_xlabel("FCFOV [deg]")
    plt.tight_layout()
    plt.savefig("../simulation-results/flat-field/normalized_intensity_fovspace.png")


n_particles = 1e8
det_size_cm = 1.408

thicknesses = [500,1250,2500]  # im um, mask thickness  # im um, mask thickness

# distance between mask and detector
distances = np.linspace(0.1 * 2 * det_size_cm, 1 * 2 * det_size_cm, 5)


all_data = []
subplot_titles = []
for thickness in thicknesses:
    datasets = []
    for di,start_distance in enumerate(distances):
        distance = round(start_distance - ((150 + (thickness / 2)) * 1e-4), 2)
        fstop = np.array(distance)/(2*det_size_cm)
        print(fstop)
        fname_tag = f"flat-field-sweep-{11}-{round(thickness)}-{di}"
        fname = f"../simulation-results/flat-field/{fname_tag}_{n_particles:.2E}_Mono_500fov.txt"
        
        data = read_two_column_file(fname)
        datasets.append(data)
    all_data.append(datasets)

    subplot_titles.append(f"{thickness} um mask")

plot_subplots(all_data, subplot_titles)


