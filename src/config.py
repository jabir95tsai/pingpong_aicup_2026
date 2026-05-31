"""Configuration constants for the ping pong prediction project."""
import os


def _has_required_data_files(path):
    if not path:
        return False
    required = ["train.csv", "sample_submission.csv"]
    has_required = all(os.path.exists(os.path.join(path, name)) for name in required)
    has_test = any(
        os.path.exists(os.path.join(path, name))
        for name in ("test_new.csv", "test.csv")
    )
    return has_required and has_test


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


def _resolve_test_path(data_dir):
    """Prefer the post-reset test_new.csv when present, with an env override."""
    env_test = os.environ.get("PINGPONG_TEST_FILE")
    if env_test:
        if os.path.isabs(env_test):
            return os.path.abspath(env_test)
        return os.path.abspath(os.path.join(data_dir, env_test))

    test_new = os.path.join(data_dir, "test_new.csv")
    if os.path.exists(test_new):
        return os.path.abspath(test_new)
    return os.path.abspath(os.path.join(data_dir, "test.csv"))


# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = _resolve_data_dir(PROJECT_ROOT)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submissions")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = _resolve_test_path(DATA_DIR)
TEST_FILE = os.path.basename(TEST_PATH)
OLD_TEST_PATH = os.path.join(DATA_DIR, "test.csv")
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
