# -*- coding: utf-8 -*-
"""
PHYS20161 Final project
@author: Pan Zhang

This is a code used for fitting breit_wigner_expression.
This code can fit three parameters or two parameters.
Generate three plot, one is fit plot, one is contour plot, and one is 3d
contour plot.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy import constants


# file read
FILE_NAME = ['z_boson_data_1.csv', 'z_boson_data_2.csv']

# picture save
SAVE_FIGURE = True # if save figure , default is True
PLOT_NAME = 'Breit_Wigner_expression_fit.png'
CONTOUR_PLOT_NAME = 'chi_squared_contour_plot.png'
CONTOUR_PLOT_3D_NAME = 'chi_squared_contour_plot_funny.png'
# file save
SAVE_FILE = True # if save data and output
SAVE_FILE_NAME = 'combine_z_boson_data.txt'# Give name to the saved file

# Name the label of fit line
DEFAULT_X_LABEL = 'Energy        Gev'
DEFAULT_Y_LABEL = 'cross_section          nb'
DEFAULT_TITLE = r'$\sigma$ against E'

# Name the label of contour plot
DEFAULT_X_CONTOUR_PLOT = 'boson mass         Gev/c^2'
DEFAULT_Y_CONTOUR_PLOT = 'boson width        Gev'
DEFAULT_CONTOUR_PLOT_TITLE = r'$\chi^2$ contours against parameter'

# adjust plot
PLOT_RANGE = 0.01 # Plot resolution
CONTOUR_PLOT_X_RANGE = 0.0007 # x range contour
CONTOUR_PLOT_Y_RANGE = 0.022 # y range contour
LEVELS_LABELS_COLORS = np.array([[0, 'Minimum', 'r'], \
                             [1, r'$\chi^2_{{\mathrm{{min.}}}}+1.00$', 'g'],\
                            [2.30, r'$\chi^2_{{\mathrm{{min.}}}}+2.30$', 'y'],\
                            [5.99, r'$\chi^2_{{\mathrm{{min.}}}}+5.99$', 'b'],\
                            [9.21, r'$\chi^2_{{\mathrm{{min.}}}}+9.21$', 'm']]
                                )
# You can choose different confidence interval by change the number from 0 to 4
# 0 is the uncertainty we want minimum chi2 + 1. 1, 2, 3 correspond to 68%, 95% and 99%
# confidence interval respectively.
DEFAULT_UNCERTAINTY = 0

# plot feature
DEFAULT_COLOR = 'black' # default color of plot layout
DELIMITER = ',' # delimiter of reading file
FMT = 'x' # The type of error bar
LINE_STYLE = 'dashed' # The format of plot line
DPI = 300 # plot resolusion
ALPHA = 0.3 # AlPHA is the transparency of the contour plot line


# Know partial width (true) represent 2 parameters fitting! Try 3 parameters fitting by switch this
# as False
KNOW_PARTIAL_WIDTH = True
PARTIAL_WIDTH = 0.08391 # The constant of partitial width in Gev

# The step will change the guess range in strong mode. Suggest range is from 0 to 1, you can
# manually change this range.
STEP = [0.2, 0.4, 0.6, 0.8, 1]
# When turn on strong mode(True), minimizing process will use the step to find out the best fit.
STRONG_MODE = False


def read_file():
    """Read file, get raw data

    Returns
    A combination of file data
    -------
    combined_data : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]

    """
    combined_data = np.zeros((0, 3))
    try:
        for name in FILE_NAME:
            file_data = np.genfromtxt(name, delimiter=DELIMITER)
            combined_data = np.vstack((combined_data, file_data))
        return combined_data
    except ValueError:
        print('Give three column data in format [x value, y value, y error'
              ']')
        sys.exit()
    except OSError:
        print('File {0:} cannot be found.'.format(name))
        sys.exit()

def detect_outlier(data_array):
    """Find outlier of raw data by statistic method, Z value.
        Turn all of the outlier into zero


    Parameters
    ----------
    data_array : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        The file data removed nan

    Returns
    -------
    detected_outlier_data : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        An array convert all outlier into zero

    """

    detected_outlier_data = np.array([])
    # Z value is normally equal to 3.
    thres_z_value = 3
    mean = np.mean(data_array)
    stdv = np.std(data_array)
    for data in data_array:
        z_score = (data-mean)/stdv
        if np.abs(z_score) > thres_z_value:
            # convert all outlier to 0 since no zero value in experiment
            data = 0
        detected_outlier_data = np.append(detected_outlier_data, data)
    return detected_outlier_data

def test_continuity(data_array, fitted_parameters):
    """ Further remove outliers if function values are not in 3 sigma interval
        around the data point.


    Parameters
    ----------
    data_array : numpy array [(float)]
        The file data removed nan and outlier
    fitted_parameters : numpy array [(float),(float),(float)]
        The fitted parameters of breit_wigner_expression. Boson mass and boson
        width and partial_width or default partial_width

    Returns
    -------
    continuous_data_point : numpy array [(float)]
        The data that allocated around three sigma range of the function
        y value.

    """
    continuous_data_point = np.zeros((0, 3))
    for line in data_array:
        fitted_y_value = breit_wigner_expression(line[0], fitted_parameters[0],
                                                 fitted_parameters[1],
                                                 fitted_parameters[2])
        if 3 * line[2] > np.abs(fitted_y_value-line[1]):
            continuous_data_point = np.vstack((continuous_data_point, line))
    return continuous_data_point

def data_filter(file_data):
    """ Integrate all remove outlier function


    Parameters
    ----------
    file_data : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        The original data in the file.

    Returns
    -------
    filtered_data : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        filtered data is the data removed outlier, nan, and
        they are continuous.

    """
    filtered_data = np.zeros((0, 3))
    filt_nan = np.isnan(file_data).any(axis=1)
    no_nan_array = ~ filt_nan
    no_nan_array = file_data[no_nan_array, :]

    data_removed_outlier_one = detect_outlier(no_nan_array[:, 0])
    data_removed_outlier_two = detect_outlier(no_nan_array[:, 1])
    data_removed_outlier_three = detect_outlier(no_nan_array[:, 2])

    temp = np.array([data_removed_outlier_one, data_removed_outlier_two,
                     data_removed_outlier_three])
    temp = np.transpose(temp)

    for entry in temp:
        if not entry[0] == 0:
            if not entry[1] == 0:
                if not entry[2] == 0:
                    filtered_data = np.vstack((filtered_data, entry))
    return filtered_data

def combined_data_file(filtered_data_array, output):
    """Save combined data as a document and output in a file


    Parameters
    ----------
    filtered_data_array : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        The completely filtered data
        saving this data in a new file
    output: list [(float)]
        summary of all routine
    Returns
    -------
    None.

    """
    combine_data_file = open(SAVE_FILE_NAME, 'w')
    data_array = np.array(['x_value', 'y_value', 'y_uncertainties'])
    output = np.array([output[0], '', ''])
    data_array = np.vstack((data_array, output))
    data_array = np.vstack((data_array, filtered_data_array))
    for data in data_array:
        combine_data_file.write(str(data[0]) + ", " +
                                str(data[1]) + ", " +
                                str(data[2]) + "\n")
    combine_data_file.close()


def breit_wigner_expression(energy, boson_mass, boson_width,
                            partial_width=PARTIAL_WIDTH):
    """A kind of distribution. cross section =
                                12pi/(Mz)^2*(E^2*(Gamma ee)^2/
                                             ((E^2-Mz^2)^2+Mz^2 * Gamma z ^2


    Parameters
    ----------
    energy : float
        The variable of this function
    boson_mass : float
        A parameter waiting for fitting
    boson_width : float
        A parameter waiting for fitting
    partial_width : float, optional
        A parameter waiting for fitting. The default is PARTIAL_WIDTH.


    Returns
    -------
    A Breit-wigner expression with two or three parameter waiting for fitting

    """

    return (0.3894 * 10**6)*(12 * np.pi/boson_mass**2) * (energy**2 *
                                                          partial_width**2)/\
                                            ((energy**2 - boson_mass**2)**2 +
                                             boson_mass**2 * boson_width**2)

def chi_squared_function(parameters, data_array):
    """ Calculate the chi squared of the given function


    Parameters
    ----------
    parameters : a tuple or list or numpy array [(float),(float)] or
                [(float),(float),(float)]
        Consist of three parameters or two
    data_array : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        The data array from file document

    Returns
    -------
    chi_square : float
        return a chi2 value based on the parameters.

    """

    if not KNOW_PARTIAL_WIDTH:
        boson_mass, boson_width, partial_width = parameters
        chi_square = np.sum(((breit_wigner_expression(data_array[:, 0],
                                                      boson_mass,
                                                      boson_width,
                                                      partial_width) -
                              data_array[:, 1])/ data_array[:, 2])**2)
    if KNOW_PARTIAL_WIDTH:
        boson_mass, boson_width = parameters
        chi_square = np.sum(((breit_wigner_expression(data_array[:, 0],
                                                      boson_mass,
                                                      boson_width,
                                                      PARTIAL_WIDTH) -
                              data_array[:, 1])/ data_array[:, 2])**2)
    return chi_square

def chi_squared_contour(boson_mass, boson_width, partial_width,
                        full_filtered_data):
    """Calculate the chi square of the given function in 2D mesh array form

    Parameters
    ----------
    boson_mass : float
        A fit parameter of breit_wigner_expression
    boson_width : float
        A fit parameter of breit_wigner_expression
    partial_width : float
        A fit parameter of breit_wigner_expression
    data_array : float
        A final filtered data array from file document

    Returns
    -------
    chi_square : mesh array
        This chi2 is different with before. Since it is a mesh array,
        it can be used for plotting contour plot.

    """

    chi_square = 0
    for entry in full_filtered_data:
        chi_square += (((breit_wigner_expression(entry[0], boson_mass,
                                                 boson_width, partial_width)
                         - entry[1]) / entry[2])**2)
    return chi_square

def minimizer(data_array):
    """minimize the chi2 to get the fitted parameters


    Parameters
    ----------
    data_array : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        data from file document

    Returns
    -------
    fitted_parameters: list [[(parameter 1),(parameter 2),(parameter 3)],
                              (minimum chi2 not reduced)]
        An list include [[fitted parameters], minimum chi2]

    """
    max_counter = len(STEP)
    if not STRONG_MODE:
        lower_boundary = np.amin(data_array[:, 0])
        upper_boundary = np.amax(data_array[:, 0])
        index_max_y_value = np.where(data_array ==
                                     np.amax(data_array[:, 1]))[0][0]
        #Based on the form of function, make a reasonable guess.

        #boson mass is the x value correspond to the peak y value
        guess_boson_mass = data_array[index_max_y_value][0]

        #boson width is the FWHH of this expression.
        guess_boson_width = np.mean(abs(lower_boundary - upper_boundary))

        #partial width should have similar order of boson width
        guess_partial_width = guess_boson_width

        if KNOW_PARTIAL_WIDTH:
            guess = (guess_boson_mass, guess_boson_width)
            temp = fmin(chi_squared_function, guess,
                        args=(data_array,), full_output=True)
            fitted_data = np.array([[temp[0][0], temp[0][1],
                                     PARTIAL_WIDTH], temp[1]])

        if not KNOW_PARTIAL_WIDTH:
            guess = (guess_boson_mass, guess_boson_width, guess_partial_width)
            temp = fmin(chi_squared_function, guess,
                        args=(data_array,), full_output=True)
            fitted_data = np.array([[temp[0][0], temp[0][1],
                                     temp[0][2]], temp[1]])

    # strong mode, return parameters that has reduced chi2 closest to 1
    if STRONG_MODE:
        counter = 0
        if KNOW_PARTIAL_WIDTH:
            reduced_chi_squared_array = np.zeros((0, 4))
        if not KNOW_PARTIAL_WIDTH:
            reduced_chi_squared_array = np.zeros((0, 5))

        while counter < max_counter:

            lower_boundary = np.amin(data_array[:, 0])
            upper_boundary = np.amax(data_array[:, 0])
            index_max_y_value = np.where(data_array ==
                                         np.amax(data_array[:, 1]))[0][0]

            guess_boson_mass = data_array[index_max_y_value][0]
            #similar guess procedure, but user can modify boson width guess
            guess_boson_width = np.mean(abs(lower_boundary -
                                            upper_boundary))* STEP[counter]
            guess_partial_width = guess_boson_width

            counter += 1

            if KNOW_PARTIAL_WIDTH:
                guess = (guess_boson_mass, guess_boson_width)
                fitted_parameters = fmin(chi_squared_function, guess,
                                         args=(data_array,), full_output=True)

                reduced_chi_squared = fitted_parameters[1] / np.abs(len(
                    data_array[:, 0] -len(fitted_parameters)))
                temp = np.array([fitted_parameters[0][0],
                                 fitted_parameters[0][1],
                                 fitted_parameters[1],
                                 abs(reduced_chi_squared - 1)])

                reduced_chi_squared_array = np.vstack((reduced_chi_squared_array
                                                       , temp))

            if not KNOW_PARTIAL_WIDTH:
                guess = (guess_boson_mass, guess_boson_width,
                         guess_partial_width)
                fitted_parameters = fmin(chi_squared_function, guess,
                                         args=(data_array,), full_output=True)
                reduced_chi_squared = fitted_parameters[1] / \
                                np.abs(len(data_array[:, 0] -
                                           len(fitted_parameters))
                                       )
                temp = np.array([fitted_parameters[0][0],
                                 fitted_parameters[0][1],
                                 fitted_parameters[0][2],
                                 fitted_parameters[1],
                                 abs(reduced_chi_squared - 1)])
                reduced_chi_squared_array = np.vstack((reduced_chi_squared_array, temp))

        if KNOW_PARTIAL_WIDTH:
            min_chi2_index = np.where(np.min(reduced_chi_squared_array[:, 3]))[0][0]
            fitted_data = reduced_chi_squared_array[min_chi2_index]
            fitted_data = np.array([[fitted_data[0],
                                     fitted_data[1],
                                     PARTIAL_WIDTH],
                                    fitted_data[2]])
        if not KNOW_PARTIAL_WIDTH:
            min_chi2_index = np.where(np.min(reduced_chi_squared_array[:, 4]))[0][0]
            fitted_data = reduced_chi_squared_array[min_chi2_index]
            fitted_data = np.array([[fitted_data[0],
                                     fitted_data[1],
                                     fitted_data[2]],
                                    fitted_data[3]])


    return fitted_data


def mesh_arrays(x_array, y_array):
    """Generate mesh array


    Parameters
    ----------
    x_array : numpy array
        x axis value, boson mass
    y_array : numpy array
        y axis value, boson width

    Returns
    -------
    x_array_mesh : numpy array
        A mesh array
    y_array_mesh : numpy array
        A mesh array

    """
    x_array_mesh = np.empty((0, len(x_array)))
    for _ in y_array:
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))
    for _ in x_array:
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)
    return x_array_mesh, y_array_mesh

def plot(data_array, fit_data, parameter_uncertainties):
    """Plot a breit wigner expression


    Parameters
    ----------
    data_array : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
    The completely filtered data point
    fit_data : numpy array [[(float),(float),(float)]
                            ,(float)]
    Three Fitted parameters
    parameter_uncertainties : numpy array [(float),(float)]
    Two fitted parameters uncertainty, boson mass and boson width.

    Returns
    -------
    None.

    """
    fit_parameters = fit_data[0]

    # main plot
    fig = plt.figure(figsize=(10, 8))
    plot_x_axis_range = np.linspace(np.amin(data_array[:, 0]) -
                                    np.mean(data_array[:, 0]) * PLOT_RANGE
                                    ,
                                    np.amax(data_array[:, 0]) +
                                    np.mean(data_array[:, 0]) * PLOT_RANGE
                                    ,
                                    1000)

    main_plot = fig.add_subplot(211)
    main_plot.scatter(data_array[:, 0], data_array[:, 1])
    main_plot.errorbar(data_array[:, 0], data_array[:, 1],
                       yerr=data_array[:, 2],
                       fmt=FMT)

    main_plot.plot(plot_x_axis_range,
                   breit_wigner_expression(plot_x_axis_range,
                                           fit_parameters[0],
                                           fit_parameters[1],
                                           fit_parameters[2]))
    # plot feature
    plt.grid()
    main_plot.set_xlabel(DEFAULT_X_LABEL, fontsize='15')
    main_plot.set_ylabel(DEFAULT_Y_LABEL, fontsize='15')
    main_plot.set_title(DEFAULT_TITLE, fontsize='17')
    # annotate, hard code
    chi_squared = fit_data[1]
    degrees_of_freedom = len(data_array[:,]) - len(fit_parameters)
    if KNOW_PARTIAL_WIDTH:
        degrees_of_freedom = degrees_of_freedom + 1
    reduced_chi_squared = chi_squared / degrees_of_freedom

    main_plot.annotate((r'$\chi^2_{{\mathrm{{min.}}}}$ = {0:.3f}'.
                        format(chi_squared)), (1, 0), (-147, -35),
                       xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')

    main_plot.annotate(('Degrees of freedom = {0:d}'.
                        format(degrees_of_freedom)), (1, 0), (-147, -60),
                       xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')

    main_plot.annotate((r'Reduced $\chi^2$ = {0:4.3f}'.
                        format(reduced_chi_squared)), (1, 0), (-147, -85),
                       xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')

    main_plot.annotate(r'$\sigma=\frac{12\pi}{m^2_{{\mathrm{{z}}}}}'
                       r' \frac{E^2 \Gamma^2_{{\mathrm{{ee}}}}}{(E^2 - '
                       r'm^2_{{\mathrm{{z}}}})^2 + m^2_{{\mathrm{{z}}}} '
                       r'\Gamma^2_{{\mathrm{{z}}}}}$', (0, 0), (0, -35),
                       xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='17')

    main_plot.annotate((r'$ m_{{\mathrm{{z}}}}$ = {0:.4g}'
                        .format(fit_parameters[0])),
                       (0, 0), (0, -75), xycoords='axes fraction',
                       va='top', textcoords='offset points', fontsize='10')
    main_plot.annotate(('± {0:.2f}'.format(parameter_uncertainties[0])),
                       (0, 0), (60, -75), xycoords='axes fraction',
                       va='top', fontsize='10',
                       textcoords='offset points')
    main_plot.annotate((r'$\Gamma_{{\mathrm{{z}}}}$ = {0:.4g}'
                        .format(fit_parameters[1])),
                       (0, 0),
                       (0, -90),
                       xycoords='axes fraction',
                       va='top', textcoords='offset points', fontsize='10')
    main_plot.annotate(('± {0:.3f}'.format(parameter_uncertainties[1])),
                       (0, 0), (60, -90), xycoords='axes fraction',
                       textcoords='offset points', va='top',
                       fontsize='10')
    main_plot.annotate((r'$\Gamma_{{\mathrm{{ee}}}}$ = {0:.4g}.'
                        .format(fit_parameters[2])),
                       (0, 0), (0, -105), xycoords='axes fraction',
                       textcoords='offset points', va='top',
                       fontsize='10')

    residuals = data_array[:, 1] - breit_wigner_expression(data_array[:, 0],
                                                           fit_parameters[0],
                                                           fit_parameters[1],
                                                           fit_parameters[2]
                                                           )
    # residual plot
    plot_residuals = fig.add_subplot(414)

    plot_residuals.errorbar(data_array[:, 0], residuals, yerr=data_array[:, 2],
                            fmt=FMT)
    plot_residuals.plot(data_array[:, 0], 0 * data_array[:, 0],
                        color=DEFAULT_COLOR)
    if SAVE_FIGURE:
        plt.savefig(PLOT_NAME, dpi=DPI)
    plt.show()

def plot_contour(fitted_data, data_array):
    """plot a contour plot
    Parameters
    ----------
    fitted_data : 2D list [[(float),(float),(float)]
                        ,(float)]
        The fitted parameters
    data_array : numpy array [(float),(float),(float)]
        The completely filtered data from file data

    Returns
    -------
    uncertainty_collection : Numpy 2D array [[(float),(float)],
                                             [(float),(float)]]
    The uncertainty of parameters

    """
    fitted_parameters = fitted_data[0]
    # generate mesh array
    x_value = np.linspace(fitted_parameters[0] - CONTOUR_PLOT_X_RANGE *
                          fitted_parameters[0], fitted_parameters[0] +
                          CONTOUR_PLOT_X_RANGE *
                          fitted_parameters[0], 300)
    y_value = np.linspace(fitted_parameters[1] - fitted_parameters[1] *
                          CONTOUR_PLOT_Y_RANGE, fitted_parameters[1] +
                          fitted_parameters[1] * CONTOUR_PLOT_Y_RANGE,
                          300)
    x_mesh, y_mesh = mesh_arrays(x_value, y_value)

    # plot feature
    parameters_contour_figure = plt.figure()
    parameters_contour_plot = parameters_contour_figure.add_subplot(111)
    parameters_contour_plot.set_title(DEFAULT_CONTOUR_PLOT_TITLE,
                                      fontsize=14)
    parameters_contour_plot.set_xlabel(DEFAULT_X_CONTOUR_PLOT, fontsize=14)
    parameters_contour_plot.set_ylabel(DEFAULT_Y_CONTOUR_PLOT, fontsize=14)
    # numpy array for capture uncertainties
    uncertainty_collection = np.zeros((0, 2))
    find_uncertainties = np.zeros((0, 6))
    labels = LEVELS_LABELS_COLORS[:, 2]

    # contour plot at uncertainties level
    for index, level_label_color in enumerate(LEVELS_LABELS_COLORS):

        level = float(level_label_color[0])
        linestyle = None
        color = level_label_color[2]

        if level == 1:
            linestyle = LINE_STYLE

        if level != 0:
            contour_level_uncertainty = parameters_contour_plot.contour(\
                                    x_mesh, y_mesh,
                                    chi_squared_contour(x_mesh, y_mesh,
                                                        fitted_parameters[2],
                                                        data_array),
                                    levels=[fitted_data[1] + level],
                                    linestyles=linestyle,
                                    colors=color)

            path = contour_level_uncertainty.collections[0].get_paths()[0]
            vertice = path.vertices

            x_value_array = vertice[:, 0]
            x_value_min = np.amin(x_value_array)
            x_value_max = np.amax(x_value_array)
            x_middle = (x_value_max - x_value_min)/2+x_value_min

            y_value_array = vertice[:, 1]
            y_value_min = np.amin(y_value_array)
            y_value_max = np.amax(y_value_array)
            y_middle = (y_value_max-y_value_min)/2 + y_value_min

            temp = np.array([x_value_min, x_value_max, x_middle, y_value_min,
                             y_value_max, y_middle])
            find_uncertainties = np.vstack((find_uncertainties, temp))

            uncertainty_boson_mass = x_value_max - x_middle
            uncertainty_boson_width = y_value_max - y_middle
            temp = np.array([uncertainty_boson_mass, uncertainty_boson_width])
            uncertainty_collection = np.vstack((uncertainty_collection, temp))

        else:
            parameters_contour_plot.scatter(fitted_parameters[0],
                                            fitted_parameters[1])

    # make sure legend do not overlap with axes
    labels = LEVELS_LABELS_COLORS[:, 1]
    box = parameters_contour_plot.get_position()
    parameters_contour_plot.set_position([box.x0, box.y0, box.width * 0.7,
                                          box.height])

    for index, label in enumerate(labels):

        parameters_contour_plot.collections[index].set_label(label)
        parameters_contour_plot.legend(loc='upper left',
                                       bbox_to_anchor=(1, 0.5), fontsize=14)

    # Plot uncertainties boundary
    entry = find_uncertainties[DEFAULT_UNCERTAINTY]
    parameters_contour_plot.scatter(entry[2], entry[3])
    parameters_contour_plot.scatter(entry[2], entry[4])
    parameters_contour_plot.scatter(entry[0], entry[5])
    parameters_contour_plot.scatter(entry[1], entry[5])
    # annotate of plot
    parameters_contour_plot.annotate(r'$\Delta m_{{\mathrm{{z}}}}$ = {0:.3g}'
                                     .format(uncertainty_collection
                                             [DEFAULT_UNCERTAINTY][0]),
                                     (1.3, 1), xycoords="axes fraction",
                                     va="center", ha="center",
                                     bbox=dict(boxstyle="round, pad=1", fc="w")
                                     )
    parameters_contour_plot.annotate(r'$\Delta \Gamma_{{\mathrm{{z}}}}$ ='
                                     '{0:.3g}'.format(uncertainty_collection
                                                      [DEFAULT_UNCERTAINTY][1]
                                                      ), (1.3, 0.85),
                                     xycoords="axes fraction",
                                     va="center", ha="center",
                                     bbox=dict(boxstyle="round, pad=1", fc="w")
                                     )
    parameters_contour_plot.annotate(r'$\Gamma_{{\mathrm{{ee}}}}$ = {0:.3g}'
                                     .format(fitted_parameters[2]),
                                     (1.3, 0.7), xycoords="axes fraction",
                                     va="center", ha="center",
                                     bbox=dict(boxstyle="round, pad=1", fc="w")
                                     )

    # main contour plot, labeled, give reference to the location of uncertainies contour plot
    contour_plot = parameters_contour_plot.contour(x_mesh, y_mesh,\
                                    chi_squared_contour(x_mesh, y_mesh,
                                                        fitted_parameters[2],
                                                        data_array), \
                                    colors=DEFAULT_COLOR, alpha=ALPHA)

    parameters_contour_plot.clabel(contour_plot)
    if SAVE_FIGURE:
        plt.savefig(CONTOUR_PLOT_NAME, dpi=DPI)
    plt.show()

    return uncertainty_collection

def plot_3d_diagram(fitted_data, data_array):
    """plot a 3d contour plot, for fun!


    Parameters
    ----------
    fitted_data : numpy array [[(float),(float),(float)]
                               ,(float)]
        fitted parameter of Breit Wigner expression and minimum chi2
    data_array : 2D numpy array [[(float),(float),(float)]
                                 [(float),(float),(float)]
                                 ...]
        Completely filtered file data

    Returns
    -------
    None.

    """

    fitted_parameters = fitted_data[0]
    # generate mesh array
    x_value = np.linspace(fitted_parameters[0] - CONTOUR_PLOT_X_RANGE
                          *fitted_parameters[0],
                          fitted_parameters[0] + CONTOUR_PLOT_X_RANGE
                          *fitted_parameters[0],
                          300)
    y_value = np.linspace(fitted_parameters[1] -
                          fitted_parameters[1] * CONTOUR_PLOT_Y_RANGE,
                          fitted_parameters[1] +
                          fitted_parameters[1] * CONTOUR_PLOT_Y_RANGE,
                          300)
    x_mesh, y_mesh = mesh_arrays(x_value, y_value)
    # main plot
    plt.figure()
    parameters_contour3d = plt.axes(projection='3d')

    parameters_contour3d.set_title(DEFAULT_CONTOUR_PLOT_TITLE)
    parameters_contour3d.set_xlabel(DEFAULT_X_CONTOUR_PLOT, fontsize=12)
    parameters_contour3d.set_ylabel(DEFAULT_Y_CONTOUR_PLOT, fontsize=14)

    parameters_contour3d.contour3D(x_mesh, y_mesh,
                                   chi_squared_contour(x_mesh,
                                                       y_mesh,
                                                       fitted_parameters[2],
                                                       data_array),
                                   cmap='binary')

    # chose a view to observe this 3D plot
    parameters_contour3d.view_init(fitted_parameters[0]/2,
                                   fitted_parameters[1]/2)
    if SAVE_FIGURE:
        plt.savefig(CONTOUR_PLOT_3D_NAME, dpi=DPI)
    plt.show()

def cal_life_time(width_gev, uncertainty_width_gev):
    """Calculate life time of boson


    Parameters
    ----------
    width_gev : float
        The width of boson in Gev

    Returns
    -------
    life_time : float
        The life time of boson in seconds
    life_time_uncertinaty_j: float
        The life time uncertainty in seconds

    """
    # convert Gev to joule
    width_j = width_gev * 1.60218 * 10**-10
    uncertainty_width_j = uncertainty_width_gev * 1.60218 * 10**-10

    life_time = constants.hbar/width_j
    life_time_uncertinaty_j = life_time * np.abs(uncertainty_width_j / width_j)

    return life_time, life_time_uncertinaty_j

def main():
    """Call each function in turn

    Returns
    -------
    return 0 if successfully run all code

    """
    #read data
    raw_data = read_file()
    filtered_data = data_filter(raw_data)
    fit_data = minimizer(filtered_data)
    final_data = test_continuity(filtered_data, fit_data[0])

    #find fitted parameters and chi2
    fit_data = minimizer(final_data)
    fit_function_parameters = fit_data[0]
    minimum_chi_square = fit_data[1]

    #get uncertainties from contour plot
    try:
        parameter_uncertainties = plot_contour(fit_data,
                                               final_data)[DEFAULT_UNCERTAINTY]
    except IndexError:
        print('choose suggested number from 0 to 3')
        sys.exit()
    plot_3d_diagram(fit_data, final_data)

    # calculate life time
    get_life_time, get_life_time_uncertainty = cal_life_time(\
        fit_function_parameters[1], parameter_uncertainties[1])
    # calculate degree of freedom
    dof = len(final_data) - len(fit_function_parameters)

    if KNOW_PARTIAL_WIDTH: # plus 1 if do 2 parameter fit
        dof = dof + 1

    output = ['Boson_mass {0:.4g} ± {3:.2f} Gev/c^2. Boson_width {1:.4g} ± '\
             '{4: .3f} Gev. The fitted partial_width is {7: .4g} Gev. Reduced'
             ' chi squared is {2:.3f} Life time of boson is '
             '{5:.3g} ± {6:.1g} seconds.'.format(fit_function_parameters[0],
                                                 fit_function_parameters[1],
                                                 minimum_chi_square/dof,
                                                 parameter_uncertainties[0],
                                                 parameter_uncertainties[1],
                                                 get_life_time,
                                                 get_life_time_uncertainty,
                                                 fit_function_parameters[2])]

    print(output[0])

    #save combined data
    if SAVE_FILE:
        combined_data_file(final_data, output)

    plot(final_data, fit_data, parameter_uncertainties)

    print('code running successful')
    return 0

if __name__ == '__main__':
    main()
