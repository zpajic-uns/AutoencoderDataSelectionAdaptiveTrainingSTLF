import numpy as np
from matplotlib import pyplot

if __name__ == '__main__':
    load_errors = np.array([4.652, 3.291, 2.669, 2.445, 1.246, 1.103, 1.039, 1.022, 1.025, 0.977, 0.914, 0.860, 0.945, 0.833, 0.888, 0.844, 0.806, 0.909, 0.891, 0.862, 0.848, 0.821, 0.876])
    temp_errors = np.array([2.323, 2.290, 1.398, 1.312, 1.304, 1.326, 1.306, 1.333, 1.321, 1.338, 1.309, 1.330, 1.304, 1.295, 1.307, 1.336, 1.304, 1.308, 1.321, 1.305, 1.328, 1.312, 1.289])
    index = np.array(range(1,24))
    print(index)

    # calculate polynomial
    z1 = np.polyfit(index, load_errors, 5)
    f1 = np.poly1d(z1)

    z2 = np.polyfit(index, temp_errors, 5)
    f2 = np.poly1d(z2)

    # calculate new x's and y's
    x_new = np.linspace(index[0], index[-1], 50)
    y_new_load = f1(x_new)
    y_new_temp = f2(x_new)
    pyplot.plot(index, load_errors, 'x', color='green', label="Load error")
    pyplot.plot(x_new, y_new_load, color='green', label="Load error - fitted curve")
    pyplot.plot(index, temp_errors, '.', color='blue', label="Air temperature error")
    pyplot.plot(x_new, y_new_temp, color='blue', label="Air temperature error - fitted curve")
    pyplot.legend(loc="upper right")
    #pyplot.title("Autoencoder model reconstruction error")
    pyplot.xlabel("Length of latent vector")
    pyplot.ylabel("Mean absolute percentage error, MAPE")

    pyplot.show()