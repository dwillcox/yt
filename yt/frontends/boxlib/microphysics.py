import re
import numpy as np
from periodictable import elements

class Nucleus(object):
    """
    a nucleus -- we store it in a
    class to hold its properties, define a sorting, and give it a
    pretty printing string
    """
    def __init__(self, name=None, field=None, omegadot_field=None):
        self.raw = name
        self.field = field # Hold the field name in the dataset
        self.omegadot_field = omegadot_field

        # element symbol and atomic weight
        if name == "p":
            self.el = "H"
            self.A = 1
            self.short_spec_name = "h1"
        elif name == "d":
            self.el = "H"
            self.A = 2
            self.short_spec_name = "h2"
        elif name == "t":
            self.el = "H"
            self.A = 3
            self.short_spec_name = "h3"
        elif name == "n":
            self.el = "n"
            self.A = 1
            self.short_spec_name = "n"
        else:
            e = re.match("([a-zA-Z]*)(\d*)", name)
            self.el = e.group(1).title()  # chemical symbol

            self.A = int(e.group(2))
            self.short_spec_name = name

        # atomic number comes from periodtable
        i = elements.isotope("{}-{}".format(self.A, self.el))
        self.Z = i.number
        self.N = self.A - self.Z

        # long name
        if i.name == 'neutron':
            self.spec_name = i.name
        else:
            self.spec_name = '{}-{}'.format(i.name, self.A)

        # latex formatted style
        self.pretty = r"{{}}^{{{}}}\mathrm{{{}}}".format(self.A, self.el)

    def __repr__(self):
        return self.raw

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.raw == other.raw

    def __lt__(self, other):
        if not self.Z == other.Z:
            return self.Z < other.Z
        else:
            return self.A < other.A

class Network(object):
    # Holds network information
    nspec  = 0
    aion   = []
    zion   = []
    aion_inv = []
    zdiva  = []
    nuclei = [] # List of Nucleus objects

    def __init__(self, nuclei=None):
        # nuclei should be a list of Nucleus objects
        if nuclei:
            self.nuclei[:] = nuclei[:]
            self.nspec = len(self.nuclei)
            self.aion  = np.array([n.A for n in self.nuclei], dtype=np.dtype('d'))
            self.zion  = np.array([n.Z for n in self.nuclei], dtype=np.dtype('d'))
            self.aion_inv = 1.0/self.aion
            self.zdiva = self.zion * self.aion_inv

    def get_nucleus_index(self, nucleus_name):
        # Given a string naming a nucleus, return the index into
        # self.nuclei where it may be found. Returns -1 if
        # the nucleus is not present.
        ntest = Nucleus(nucleus_name)
        for i, ni in enumerate(self.nuclei):
            if ni == ntest:
                return i
        return -1

class Microphysics(object):
    # Holds dataset attributes relevant to its microphysics
    network = None

    def __init__(self, field_info_container=None, nuclei=None):
        # nuclei should be a list of Nucleus objects
        self.network = Network(nuclei)

        # create derived fields in the supplied field_info_container
        # add mass fractions field
        func = self._create_massfrac_func()
        field_info_container.add_field(name=("microphysics", "mass_fractions"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'xn')

        # Check to see if each Nucleus has an omegadot field
        # and if so add the omegadots derived field.
        omegadot_present = True
        for n in self.network.nuclei:
            if not n.omegadot_field:
                omegadot_present = False
                break
        if omegadot_present:
            # create an omegadots field analogous to mass fractions field
            func = self._create_omegadots_func()
            field_info_container.add_field(name=("microphysics", "omegadots"),
                                           sampling_type="cell",
                                           function=func,
                                           units="1/s",
                                           display_name=r'omegadot')

        # add helper fields
        func = self._create_sum_xdiva_func()
        field_info_container.add_field(name=("microphysics", "sum_xdiva"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'sum(xn/aion)')

        # add electron fraction, abar, and zbar fields
        func = self._create_abar_func()
        field_info_container.add_field(name=("microphysics", "abar"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'abar')

        func = self._create_electron_fraction_func()
        field_info_container.add_field(name=("microphysics", "electron_fraction"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'ye')

        func = self._create_zbar_func()
        field_info_container.add_field(name=("microphysics", "zbar"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'zbar')

    def _create_massfrac_func(self):
        # Returns a YTArray object containing the mass fractions
        # Dimensions of the array are: [species], [spatial], ([spatial], ...)
        # That is, the first axis corresponds to species, the
        # second axis corresponds to a spatial index, and subsequent axes
        # may also correspond to spatial indices depending on the
        # dimensionality of the dataset.
        def _func(field, data):
            return data.ds.arr([data[n.field] for n in self.network.nuclei], "")
        return _func

    def _create_omegadots_func(self):
        # Returns a YTArray object containing the mass fractions
        # Dimensions of the array are: [species], [spatial], ([spatial], ...)
        # That is, the first axis corresponds to species, the
        # second axis corresponds to a spatial index, and subsequent axes
        # may also correspond to spatial indices depending on the
        # dimensionality of the dataset.
        def _func(field, data):
            return data.ds.arr([data[n.omegadot_field] for n in self.network.nuclei], "")
        return _func

    def _create_sum_xdiva_func(self):
        def _func(field, data):
            xn   = data[('microphysics', 'mass_fractions')]
            aion_inv = data.ds.arr(data.ds.microphysics.network.aion_inv)
            # Form a dot product between aion_inv and the species axis of xn
            sum_xdiva = np.tensordot(aion_inv, xn, axes=([0], [0]))
            return sum_xdiva
        return _func

    def _create_abar_func(self):
        def _func(field, data):
            xn   = data[('microphysics', 'mass_fractions')]
            sum_xdiva = data[('microphysics', 'sum_xdiva')]
            # Sum over species axis of xn
            sum_x = np.sum(xn, axis=0)
            abar = sum_x/sum_xdiva
            return abar
        return _func

    def _create_electron_fraction_func(self):
        def _func(field, data):
            xn = data[('microphysics', 'mass_fractions')]
            zdiva = data.ds.arr(data.ds.microphysics.network.zdiva)
            # Form a dot product between zdiva and the species axis of xn
            ye = np.tensordot(zdiva, xn, axes=([0], [0]))
            return ye
        return _func

    def _create_zbar_func(self):
        def _func(field, data):
            ye   = data[('microphysics', 'electron_fraction')]
            sum_xdiva = data[('microphysics', 'sum_xdiva')]
            zbar = ye/sum_xdiva
            return zbar
        return _func
