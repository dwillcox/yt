import re
import numpy as np
from periodictable import elements

class Nucleus(object):
    """
    a nucleus -- we store it in a
    class to hold its properties, define a sorting, and give it a
    pretty printing string
    """
    def __init__(self, name=None, specA=-1, specZ=-1, field=None, omegadot_field=None):
        # specA and specZ are used to set A, Z, N for a species if it cannot be found
        # in the periodictable based on the name argument. This allows for networks
        # that specify e.g. an "ash" species that might represent a mixture of various
        # nuclei not carried individually in the network. specA or specZ may thus be floats.

        self.raw = name

        # Hold the field name in the dataset and its omegadot,
        # if provided.
        self.field = field
        self.omegadot_field = omegadot_field

        is_physical_species = False

        # element symbol and atomic weight
        if name == "p":
            self.el = "H"
            self.A = 1
            self.short_spec_name = "h1"
            is_physical_species = True
        elif name == "d":
            self.el = "H"
            self.A = 2
            self.short_spec_name = "h2"
            is_physical_species = True
        elif name == "t":
            self.el = "H"
            self.A = 3
            self.short_spec_name = "h3"
            is_physical_species = True
        elif name == "n":
            self.el = "n"
            self.A = 1
            self.short_spec_name = "n"
            is_physical_species = True
        else:
            e = re.match("([a-zA-Z]*)(\d*)", name)
            if e:
                self.el = e.group(1).title()  # chemical symbol

                self.A = int(e.group(2))
                self.short_spec_name = name
                is_physical_species = True
            else:
                # we could not identify a nuclear species, so create a Nucleus
                # with A and Z (and N) set by specA and specZ, if supplied.
                # If specA and specZ are not supplied, this is an error.
                if (specA >= 0 and specZ >= 0):
                    is_physical_species = False
                    self.el = self.raw
                    self.A = float(specA)
                    self.Z = float(specZ)
                    self.N = self.A - self.Z
                else:
                    raise RuntimeError("microphysics: did not recognize species {}.".format(name))

        if is_physical_species:
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
        else:
            # latex formatted style
            self.pretty = r"\mathrm{{{}}}".format(self.raw)

    def set_field(self, field):
        self.field = field

    def set_omegadot_field(self, omegadot_field):
        self.omegadot_field = omegadot_field

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

    def __contains__(self, nucleus_name):
        inuc = self._get_nucleus_index(nucleus_name, silent_failure=True)
        if inuc != -1:
            return True
        else:
            return False

    def nucleus(self, nname):
        # Return the Nucleus object corresponding to the nucleus name nname
        idx = self._get_nucleus_index(nname, silent_failure=True)
        if idx != -1:
            return self.nuclei[idx]
        else:
            return None

    def _get_nucleus_index(self, name, silent_failure=False):
        # Given a string naming a nucleus, return the index into
        # self.nuclei where it may be found. Returns -1 if
        # the nucleus is not present.
        for i, ni in enumerate(self.nuclei):
            if ni.raw == name:
                return i
        if silent_failure:
            return -1
        else:
            raise RuntimeError("microphysics: species {} could not be found in network.".format(name))

    def set_field(self, nucleus_name, field_name):
        inuc = self._get_nucleus_index(nucleus_name)
        self.nuclei[inuc].set_field(field_name)

    def set_omegadot_field(self, nucleus_name, omegadot_field_name):
        inuc = self._get_nucleus_index(nucleus_name)
        self.nuclei[inuc].set_omegadot_field(omegadot_field_name)

class Microphysics(object):
    # Holds dataset attributes relevant to its microphysics
    network = None

    # Holds a list of fields particular to the Microphysics class
    field_list = []

    def __init__(self, field_info_container=None, nuclei=None):
        if nuclei:
            self.setup_network(nuclei)
        if field_info_container:
            self.setup_fields(field_info_container)

    def setup_network(self, nuclei=None):
        # nuclei should be a list of Nucleus objects
        self.network = Network(nuclei)

    def setup_fields(self, field_info_container=None):
        # create derived fields in the supplied field_info_container
        # add mass fractions field
        func = self._create_massfrac_func()
        field_info_container.add_field(name=("microphysics", "mass_fractions"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'xn')
        self.field_list.append(("microphysics", "mass_fractions"))

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
            self.field_list.append(("microphysics", "omegadots"))

        # add helper fields
        func = self._create_sum_xdiva_func()
        field_info_container.add_field(name=("microphysics", "sum_xdiva"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'sum(xn/aion)')
        self.field_list.append(("microphysics", "sum_xdiva"))

        # add electron fraction, abar, and zbar fields
        func = self._create_abar_func()
        field_info_container.add_field(name=("microphysics", "abar"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'abar')
        self.field_list.append(("microphysics", "abar"))

        func = self._create_electron_fraction_func()
        field_info_container.add_field(name=("microphysics", "electron_fraction"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'ye')
        self.field_list.append(("microphysics", "electron_fraction"))

        func = self._create_zbar_func()
        field_info_container.add_field(name=("microphysics", "zbar"),
                                       sampling_type="cell",
                                       function=func,
                                       units="",
                                       display_name=r'zbar')
        self.field_list.append(("microphysics", "zbar"))

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
