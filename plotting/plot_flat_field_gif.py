from PIL import Image
import glob

# List of image file paths
# img_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/geom-correction/"
img_folder = "/home/rileyannereid/workspace/geant4/simulation-results/rings/"

# Search for PNG files containing "dc" and "-y-" in the filename
image_paths = sorted(
    [file for file in glob.glob(f"{img_folder}/61-2-*-d3-2p25-scale_*_dc.png")]
)

image_paths = [item for item in image_paths if "test" not in item]


def custom_sorting_key(s):
    # Split each string by "-" and convert the third element (index 2) to an integer
    # return int(s.split("-")[5])
    parts = s.split("-")
    third_part = parts[3].split("p")[0]
    return float(third_part)


# Sort the list based on the custom sorting key
sorted_list = sorted(image_paths[::2], key=custom_sorting_key)

# first_img = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/geom-correction/61-2-0-0-geom-corr-0-ptsrc_1.00E+06_Mono_100_dc.png"
# sorted_list.insert(0, first_img)
# Open and convert images to GIF
images = [Image.open(img) for img in sorted_list]

# Save the GIF
images[0].save(
    f"{img_folder}d3-2p25-rings-scale.gif",
    save_all=True,
    append_images=images[1:],
    duration=1000,
    loop=0,
)
