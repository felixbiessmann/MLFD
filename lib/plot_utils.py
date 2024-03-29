import math
from pathlib import Path
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile


def get_fig_size(fig_width_cm, fig_height_cm=None):
    """Convert dimensions from centimeters to inches.
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_height_cm:
        golden_ratio = (1 + math.sqrt(5))/2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return map(lambda x: x/2.54, size_cm)


"""
The following functions can be used by scripts to get the sizes of
the various elements of the figures.
"""


def label_size():
    """Size of axis labels """
    return 6  # 10pt is latex standard font-size


def font_size():
    """Size of all texts shown in plots """
    return 6


def ticks_size():
    """Size of axes' ticks' labels """
    return 6


def axis_lw():
    """Line width of the axes """
    return 0.6


def plot_lw():
    """Line width of the plotted curves """
    return 1.5


def figure_setup():
    """
    Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {'text.usetex': True,
              'figure.dpi': 200,
              'font.size': font_size(),
              'font.serif': 'Palatino',
              'font.sans-serif': 'Palatino',
              'font.monospace': 'Palatino',
              'axes.labelsize': label_size(),
              'axes.titlesize': font_size(),
              'axes.linewidth': axis_lw(),
              'legend.fontsize': font_size(),
              'xtick.labelsize': ticks_size(),
              'ytick.labelsize': ticks_size(),
              'font.family': 'serif'}

    # uncomment if the thesis LATEX project is in MLFD's the root folder
    #plt.style.use('thesis')

    plt.rcParams.update(params)


def save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """
    Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.

    Keyword Attributes:
    fig -- Pyplot fig-Object ready to be saved
    file_name -- Name under which figure will be saved
    fmt -- filetype, either eps, png or pdf are valid
    dpi -- Dots per Inch
    tight -- Boolean, whether or not figure shall be exported tight=True
    parameter
    """

    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # check if directory exists
    # Path().mkdir(parents=True, exist_ok=True)

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('cp %s %s' % (tmp_name, file_name), shell=True)
