


class SelfLearningParams():

    def __init__(self, n_round_teacher_model: int, confidence_threshold: float) -> None:
        self._n_round_teacher_model = n_round_teacher_model
        self._confidence_threshold = confidence_threshold

    @property
    def n_round_teacher_model(self):
        return self._n_round_teacher_model
    
    @property
    def confidence_threshold(self):
        return self._confidence_threshold