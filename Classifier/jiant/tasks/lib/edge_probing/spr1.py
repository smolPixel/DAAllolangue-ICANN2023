"""Semantic proto-role (1) Edge Probing task.

Task source paper: https://arxiv.org/pdf/1905.06316.pdf.
Task data prep directions: https://github.com/nyu-mll/Classifier.jiant/blob/master/probing/data/README.md.

"""
from dataclasses import dataclass

from Classifier.jiant.tasks.lib.templates.shared import labels_to_bimap
from Classifier.jiant.tasks.lib.templates import edge_probing_two_span
from Classifier.jiant.utils.python.io import read_json_lines


@dataclass
class Example(edge_probing_two_span.Example):
    @property
    def task(self):
        return Spr1Task


@dataclass
class TokenizedExample(edge_probing_two_span.TokenizedExample):
    pass


@dataclass
class DataRow(edge_probing_two_span.DataRow):
    pass


@dataclass
class Batch(edge_probing_two_span.Batch):
    pass


class Spr1Task(edge_probing_two_span.AbstractProbingTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    LABELS = [
        "awareness",
        "change_of_location",
        "change_of_state",
        "changes_possession",
        "created",
        "destroyed",
        "existed_after",
        "existed_before",
        "existed_during",
        "exists_as_physical",
        "instigation",
        "location_of_event",
        "makes_physical_contact",
        "manipulated_by_another",
        "predicate_changed_argument",
        "sentient",
        "stationary",
        "volition",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 2

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (line_num, line) in enumerate(lines):
            # A line in the task's data file can contain multiple targets (span-pair + labels).
            # We create an example for every target:
            for (target_num, target) in enumerate(line["targets"]):
                span1 = target["span1"]
                span2 = target["span2"]
                examples.append(
                    Example(
                        guid="%s-%s-%s" % (set_type, line_num, target_num),
                        text=line["text"],
                        span1=span1,
                        span2=span2,
                        labels=target["label"] if set_type != "test" else [cls.LABELS[-1]],
                    )
                )
        return examples
