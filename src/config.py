"""Configuration constants for the ping pong prediction project."""
import os


def _has_required_data_files(path):
    if not path:
        return False
    required = ["train.csv", "test.csv", "sample_submission.csv"]
    return all(os.path.exists(os.path.join(path, name)) for name in required)


def _resolve_data_dir(project_root):
    """Pick first valid data dir: env override -> project/data -> hardcoded fallback."""
    candidates = []
    env_dir = os.environ.get("PINGPONG_DATA_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append(os.path.join(project_root, "data"))
    for candidate in candidates:
        if _has_required_data_files(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(project_root, "data"))


# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = _resolve_data_dir(PROJECT_ROOT)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submissions")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# Column definitions
ID_COL = "rally_uid"
TARGET_ACTION = "actionId"
TARGET_POINT = "pointId"
TARGET_SERVER = "serverGetPoint"

FEATURE_COLS = [
    "sex", "numberGame", "rally_id", "strikeNumber",
    "scoreSelf", "scoreOther",
    "gamePlayerId", "gamePlayerOtherId",
    "strikeId", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId",
]

CATEGORICAL_STRIKE_COLS = ["strikeId", "handId", "strengthId", "spinId",
                           "pointId", "actionId", "positionId"]

# Class counts
N_ACTION_CLASSES = 19  # 0-18
N_POINT_CLASSES = 10   # 0-9

# Action categories
ACTION_ATTACK = {1, 2, 3, 4, 5, 6, 7}
ACTION_CONTROL = {8, 9, 10, 11}
ACTION_DEFENSE = {12, 13, 14}
ACTION_SERVE = {15, 16, 17, 18}

# Rules / constraints
SERVE_ACTION_IDS = {0, 15, 16, 17, 18}  # actionId when strikeId=1
RETURN_FORBIDDEN_ACTIONS = {15, 16, 17, 18}  # cannot appear on return

# Lag features config
LAG_STEPS = [1, 2, 3, 5]
LAG_COLS = ["actionId", "pointId", "handId", "strengthId", "spinId", "positionId", "strikeId"]

# Model config
N_FOLDS = 5
RANDOM_SEED = 42
