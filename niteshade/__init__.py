# Automatically set the version based on the tag number
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Expose all modules
from .data import *
from .models import *
from .attack import *
from .defence import *
from .postprocessing import *
from .simulation import *
from .utils import *
