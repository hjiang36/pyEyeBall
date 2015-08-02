from Objects.Illuminant import Illuminant

__author__ = 'HJ'


class Scene:
    """
    Class describing scene radiance
    """
    name = "Scene"     # name of the object
    photons = None     # scene photons
    wave = None        # wavelength sampling in nm
    fov = None         # field of view of the scene in degree
    distance = None    # viewing distance in meters
    illuminant = None  # illuminant

    def __init__(self):
        """
        Class constructor, initializing parameters
        :return: class object with properties initialized
        """
        self.illuminant = Illuminant()
