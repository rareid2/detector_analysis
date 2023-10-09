import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


hex_list = [
    "#0091ad",
    "#3fcdda",
    "#83f9f8",
    "#d6f6eb",
    "#fdf1d2",
    "#f8eaad",
    "#faaaae",
    "#FFFFFF",
]
# hex_list = ["#%s" % hl for hl in hex_list]
# hex_list = ['#8ECAEE6','#219EBC','#023047']

# prospectus colors
# hex_list = ["#023047","#219EBC","#FFB703","#FB8500","#F15025"]
hex_list = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
hex_list = ["#ffbe0b", "#fb5607", "#ff006e", "#8338ec", "#3a86ff"]
hex_list = ['#05668d', '#028090', '#00a896', '#02c39a', '#f0f3bd']
hex_list = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]
cmap = get_continuous_cmap(hex_list)

hex_colors = []
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    hex_colors.append(mcolors.rgb2hex(rgba))
