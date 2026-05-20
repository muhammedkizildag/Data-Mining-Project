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
        ["-m", "preprocessing.preprocess_dropout"],
        ["-m", "preprocessing.prepare_oulad"],
    ],
    "train": [
        ["-m", "modeling.model_dropout_localized"],
        ["-m", "modeling.model_oulad_v2"],
    ],
    "shap": [
        ["-m", "modeling.shap_dropout_localized"],
        ["-m", "modeling.shap_oulad"],
    ],
    "chatbot-prep": [["-m", "chatbot.prepare_chatbot"]],
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
