from typing import Dict
import numpy as np


class Metrics:
    def __init__(self, n_classes, name):
        super().__init__()
        self.n_classes = n_classes
        self.name = name
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.results = {}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0


class StreamClsMetrics(Metrics):

    def __init__(self, n_classes, name):
        super().__init__(n_classes=n_classes, name=name)

    def update(self, label, prediction):
        self.confusion_matrix[label[0]][prediction[0]] += 1
        self.total_samples += 1

    def get_results(self):
        eps = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + eps)
        acc_cls = np.mean(acc_cls_c[mask])
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        self.results = {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Class Acc": cls_acc
        }

    def __str__(self):
        string = "\n"
        ignore = ["Class Acc", "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in self.results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)
        string += 'Class Acc:\n'
        for k, v in self.results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))
        return string


class StreamSegMetrics(Metrics):

    def __init__(self, n_classes, name):
        super().__init__(n_classes=n_classes, name=name)

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def get_results(self):

        eps = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + eps)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diag / (hist.sum(axis=0) + eps)
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + eps)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prec = dict(zip(range(self.n_classes), [precision_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        self.results = {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Mean Precision": precision_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
            "Class Prec": cls_prec,
        }

        return self.results

    def confusion_matrix_to_text(self):
        string = []
        for i in range(self.n_classes):
            string.append(f"{i} : {self.confusion_matrix[i].tolist()}")
        return "\n" + "\n".join(string)

    def __str__(self):
        if len(self.results) == 0:
            self.get_results()

        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in self.results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in self.results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in self.results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Prec:\n'
        for k, v in self.results['Class Prec'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string
    
class AggregatedFederatedMetrics():

    def __init__(self, name: str) -> None:
        self.name = name
        self.n_clients = 0
        self.results = {}
        self.clients_result = {}

    def update(self, metric: StreamSegMetrics, client_name: str) -> None:
        update = metric.get_results()
        self.clients_result[client_name] = update
        self.n_clients += 1
        updates_to_save = ["Overall Acc", "Mean Acc", "Mean Precision", "Mean IoU"]
        for key in updates_to_save:
            self.results[key] = self.results[key] + update[key] \
                if key in self.results.keys() else update[key]

    def calculate_results(self) -> Dict[str, float]:
        results = {}
        for k, v in self.results.items():
            results[k] = v / self.n_clients
        return results
    
    def __str__(self) -> str:
        results = self.calculate_results()
        string = "\n"
        for k, v in results.items():
            string += "%s: %f\n" % (k, v)
        for k, v in self.clients_result.items():
            string += f"{k}:\n"
            string += "Mean IoU: %f\n" % (v["Mean IoU"])
            string += "Mean Acc: %f\n" % (v["Mean Acc"])
        return string

    
