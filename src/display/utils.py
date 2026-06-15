from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.about import Tasks


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False


## Leaderboard columns
# Built as a plain class with ColumnContent attributes instead of
# make_dataclass, which breaks on Python 3.13+ due to mutable defaults.
#
# Column groups (order = display order):
#   1) Identity:  rank (T), Model, Model Family
#   2) Overall:   Average ⬆️
#   3) Trends:    1-Day, 3-Day Avg, 7-Day Avg
#   4) Per-task:  12 task category accuracies (added dynamically below)
class AutoEvalColumn:
    # --- Identity ---
    rank = ColumnContent("T", "number", True, never_hidden=True)
    model = ColumnContent("Model", "markdown", True, never_hidden=True)
    # model_family is kept in the DataFrame so the filter still works,
    # but hidden from the visible table and the column-selector list.
    model_family = ColumnContent("Model Family", "str", False, hidden=True)
    # --- Overall score ---
    average = ColumnContent("Average ⬆️", "number", True)
    # --- Trend columns (1-day, 3-day, 7-day rolling averages) ---
    one_day = ColumnContent("1-Day", "number", True)
    three_day_avg = ColumnContent("3-Day Avg", "number", True)
    seven_day_avg = ColumnContent("7-Day Avg", "number", True)


# --- Per-task category columns (12) ---
for _task in Tasks:
    setattr(AutoEvalColumn, _task.name, ColumnContent(_task.value.col_name, "number", True))


## For the queue columns in the submission tab
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    private = ColumnContent("private", "bool", True)
    precision = ColumnContent("precision", "str", True)
    weight_type = ColumnContent("weight_type", "str", "Original")
    status = ColumnContent("status", "str", True)


## All the model information that we might need
@dataclass
class ModelDetails:
    name: str
    display_name: str = ""
    symbol: str = ""  # emoji


class ModelType(Enum):
    PT = ModelDetails(name="pretrained", symbol="🟢")
    FT = ModelDetails(name="fine-tuned", symbol="🔶")
    IFT = ModelDetails(name="instruction-tuned", symbol="⭕")
    RL = ModelDetails(name="RL-tuned", symbol="🟦")
    Unknown = ModelDetails(name="", symbol="?")

    def to_str(self, separator=" "):
        return f"{self.value.symbol}{separator}{self.value.name}"

    @staticmethod
    def from_str(type):
        if "fine-tuned" in type or "🔶" in type:
            return ModelType.FT
        if "pretrained" in type or "🟢" in type:
            return ModelType.PT
        if "RL-tuned" in type or "🟦" in type:
            return ModelType.RL
        if "instruction-tuned" in type or "⭕" in type:
            return ModelType.IFT
        return ModelType.Unknown


class WeightType(Enum):
    Adapter = ModelDetails("Adapter")
    Original = ModelDetails("Original")
    Delta = ModelDetails("Delta")


class Precision(Enum):
    float16 = ModelDetails("float16")
    bfloat16 = ModelDetails("bfloat16")
    Unknown = ModelDetails("?")

    def from_str(precision):
        if precision in ["torch.float16", "float16"]:
            return Precision.float16
        if precision in ["torch.bfloat16", "bfloat16"]:
            return Precision.bfloat16
        return Precision.Unknown


# Column selection — keep ALL columns in the DataFrame (hidden ones are
# still needed for filtering/search). The Leaderboard component will
# remove `hidden=True` ones from the visible table via `hide_columns`.
COLS = [c.name for c in fields(AutoEvalColumn)]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Tasks]
