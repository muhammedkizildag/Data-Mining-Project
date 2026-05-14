import argparse
import os
import subprocess
import sys


COMMANDS = {
    "eda": [
        ["eda/eda_dropout.py"],
        ["eda/eda_oulad.py"],
        ["eda/eda_habits.py"],
    ],
    "preprocess": [
        ["preprocessing/preprocess_dropout.py"],
        ["preprocessing/prepare_oulad.py"],
    ],
    "train": [
        ["modeling/model_dropout_localized.py"],
        ["modeling/model_oulad_v2.py"],
    ],
    "shap": [
        ["modeling/shap_dropout_localized.py"],
        ["modeling/shap_oulad.py"],
    ],
    "chatbot-prep": [["chatbot/prepare_chatbot.py"]],
    "test": [["-m", "unittest", "discover", "-s", "tests", "-v"]],
}


def run_group(name):
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", ".matplotlib")
    for args in COMMANDS[name]:
        subprocess.run([sys.executable, *args], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Project task runner for preprocessing, training, and tests."
    )
    parser.add_argument(
        "task",
        choices=sorted(COMMANDS.keys()),
        help="Task group to run.",
    )
    args = parser.parse_args()
    run_group(args.task)


if __name__ == "__main__":
    main()
