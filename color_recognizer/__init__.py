import logging
import io
import urllib

import voluptuous as vol

from homeassistant.helpers import config_validation as cv
from homeassistant.const import (ATTR_ENTITY_ID, SERVICE_TURN_ON)
from homeassistant.components.light import (ATTR_RGB_COLOR)
from homeassistant.components import light

REQUIREMENTS = ['numpy==1.16', 'pillow']

_LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_RESIZE = (100, 100)
ATTR_URL = 'url'
SERVICE_RECOGNIZE_COLOR_AND_SET_LIGHT = 'turn_light_to_recognized_color'
DOMAIN = 'color_recognizer'

RECOGNIZE_COLOR_SCHEMA = vol.Schema({
    vol.Required(ATTR_URL): cv.url,
    vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
}, extra=vol.ALLOW_EXTRA)


def setup(hass, config):
    def turn_lights_to_recognized_color(call):
        call_data = dict(call.data)
        colors = ColorRecognizer(hass, config[DOMAIN], call_data.pop(ATTR_URL)).best_colors()
        call_data.update({ATTR_RGB_COLOR: colors})

        hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call_data)

    hass.services.register(DOMAIN, SERVICE_RECOGNIZE_COLOR_AND_SET_LIGHT, turn_lights_to_recognized_color, schema=RECOGNIZE_COLOR_SCHEMA)

    return True


""" Taken from: https://github.com/davidkrantz/Colorfy """
class SpotifyBackgroundColor:
    """Analyzes an image and finds a fitting background color.

    Main use is to analyze album artwork and calculate the background
    color Spotify sets when playing on a Chromecast.

    Attributes:
        img (ndarray): The image to analyze.

    """

    def __init__(self, img, format='RGB', image_processing_size=None):
        """Prepare the image for analyzation.

        Args:
            img (ndarray): The image to analyze.
            format (str): Format of `img`, either RGB or BGR.
            image_processing_size: (int/float/tuple): Process image or not.
                int - Percentage of current size.
                float - Fraction of current size.
                tuple - Size of the output image (must be integers).

        Raises:
            ValueError: If `format` is not RGB or BGR.

        """
        import numpy as np
        from PIL import Image

        img = np.array(Image.open(img))

        if format == 'RGB':
            self.img = img
        elif format == 'BGR':
            self.img = self.img[..., ::-1]
        else:
            raise ValueError('Invalid format. Only RGB and BGR image ' \
                             'format supported.')

        if image_processing_size:
            self.img = np.array(Image.fromarray(self.img).resize(image_processing_size, resample=Image.BILINEAR))

        self.img = self.img.astype(float)

    def best_color(self, k=8, color_tol=10):
        """Returns a suitable background color for the given image.

        Uses k-means clustering to find `k` distinct colors in
        the image. A colorfulness index is then calculated for each
        of these colors. The color with the highest colorfulness
        index is returned if it is greater than or equal to the
        colorfulness tolerance `color_tol`. If no color is colorful
        enough, a gray color will be returned. Returns more or less
        the same color as Spotify in 80 % of the cases.

        Args:
            k (int): Number of clusters to form.
            color_tol (float): Tolerance for a colorful color.
                Colorfulness is defined as described by Hasler and
                Süsstrunk (2003) in https://infoscience.epfl.ch/
                record/33994/files/HaslerS03.pdf.
            plot (bool): Plot the original image, k-means result and
                calculated background color. Only used for testing.

        Returns:
            tuple: (R, G, B). The calculated background color.

        """
        import numpy as np

        self.img = self.img.reshape((self.img.shape[0]*self.img.shape[1], 3))

        centroids = k_means(self.img, k)

        colorfulness = [self.colorfulness(color[0], color[1], color[2]) for color in centroids]
        max_colorful = np.max(colorfulness)

        if max_colorful < color_tol:
            # If not colorful, set to gray
            best_color = [230, 230, 230]
        else:
            # Pick the most colorful color
            best_color = centroids[np.argmax(colorfulness)]

        return int(best_color[0]), int(best_color[1]), int(best_color[2])

    def colorfulness(self, r, g, b):
        """Returns a colorfulness index of given RGB combination.

        Implementation of the colorfulness metric proposed by
        Hasler and Süsstrunk (2003) in https://infoscience.epfl.ch/
        record/33994/files/HaslerS03.pdf.

        Args:
            r (int): Red component.
            g (int): Green component.
            b (int): Blue component.

        Returns:
            float: Colorfulness metric.

        """
        import numpy as np

        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)

        # Compute the mean and standard deviation of both `rg` and `yb`.
        rb_mean, rb_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))

        # Combine the mean and standard deviations.
        std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))

        return std_root + (0.3 * mean_root)


class ColorRecognizer:
    def __init__(self, hass, component_config, url):
        self.hass = hass
        self.config = component_config
        self.url = url

    def best_colors(self):
        image = self.download_image()
        return SpotifyBackgroundColor(image, image_processing_size=DEFAULT_IMAGE_RESIZE).best_color(k=4, color_tol=5)

    def download_image(self):
        return io.BytesIO(urllib.request.urlopen(self.url).read())

def k_means(data, k=2, max_iter=100):
    """Assigns data points into clusters using the k-means algorithm.

    Parameters
    ----------
    data : ndarray
        A 2D array containing data points to be clustered.
    k : int, optional
        Number of clusters (default = 2).
    max_iter : int, optional
        Number of maximum iterations

    Returns
    -------
    labels : ndarray
        A 1D array of labels for their respective input data points.
    """

    import numpy as np

    # data_max/data_min : array containing column-wise maximum/minimum values
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    n_samples = data.shape[0]
    n_features = data.shape[1]

    # labels : array containing labels for data points, randomly initialized
    labels = np.random.randint(low=0, high=k, size=n_samples)
    # centroids : 2D containing centroids for the k-means algorithm
    # randomly initialized s.t. data_min <= centroid < data_max
    centroids = np.random.uniform(low=0., high=1., size=(k, n_features))
    centroids = centroids * (data_max - data_min) + data_min

    # k-means algorithm
    for i in range(max_iter):
        # distances : between datapoints and centroids
        distances = np.array(
            [np.linalg.norm(data - c, axis=1) for c in centroids])
        # new_labels : computed by finding centroid with minimal distance
        new_labels = np.argmin(distances, axis=0)

        if (labels == new_labels).all():
            # labels unchanged
            labels = new_labels
            # print('Labels unchanged ! Terminating k-means.')
            break
        else:
            # labels changed
            # difference : percentage of changed labels
            difference = np.mean(labels != new_labels)
            # print('%4f%% labels changed' % (difference * 100))
            labels = new_labels
            for c in range(k):
                # computing centroids by taking the mean over associated data points
                centroids[c] = np.mean(data[labels == c], axis=0)

    return labels
