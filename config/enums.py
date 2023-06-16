import enum

class DatasetOptions(str, enum.Enum):
    IDDA = "idda"
    GTA = "gta"
    IDDA_SELF = "idda_sl"
    IDDA_SILO = "idda_silo"

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
    FDA = "fda-training"
    SL = "self-learning"
    SILO = "silo-self-learning"
    BASIC_SILO = "basic-silo-self-learning"
    DEBUG = "debug"
    EXAM = "exam-project"

class NormOptions(str, enum.Enum):
    EROS = "eros_norm"
    CTS = "cts_norm"
    GTA = "gta_norm"

class ExperimentPhase(str, enum.Enum):
    ALL = "all"
    TRAIN = "train"
    TEST = "test"