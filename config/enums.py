import enum

class DatasetOptions(str, enum.Enum):
    IDDA = "idda"

class ModelOptions(str, enum.Enum):
    DEEPLABv3_MOBILENETv2 = "deeplabv3_mobilenetv2"

class OptimizerOptions(str, enum.Enum):
    SGD = "SGD"
    ADAM = "Adam"

class SchedulerOptions(str, enum.Enum):
    POLY = "poly"
    STEP = "step"

class ProjectNameOptions(str, enum.Enum):
    CENTR = "centralized-training"
    FED = "federated-training"
    DEBUG = "debug"

class NormOptions(str, enum.Enum):
    EROS = "eros_norm"
    CTS = "cts_norm"

class ExperimentPhase(str, enum.Enum):
    ALL = "all"
    TRAIN = "train"
    TEST = "test"