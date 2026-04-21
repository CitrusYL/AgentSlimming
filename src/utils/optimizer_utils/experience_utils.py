from collections import defaultdict
from pathlib import Path

from src.optimizer.workspace import WorkspaceStore, WorkflowDirs
from src.utils.logs import logger
from src.utils.common import read_json_file


class ExperienceUtils:
    def __init__(self, root_path: str, workflow_name: str = WorkflowDirs.GRAPH):
        self.root_path = root_path
        self.store = WorkspaceStore(root_path)
        self.workflow_name = workflow_name

    def load_experience(self, workflow_path: str | Path | None = None):
        rounds_dir = self._workflow_root(workflow_path)

        experience_data = defaultdict(lambda: {"score": None, "success": {}, "failure": {}})
        if not rounds_dir.is_dir():
            return dict(experience_data)

        for round_path in rounds_dir.iterdir():
            if round_path.is_dir() and round_path.name.startswith("round_"):
                try:
                    round_number = int(round_path.name.split("_")[1])
                    json_file_path = round_path / "experience.json"
                    if json_file_path.exists():
                        data = read_json_file(str(json_file_path), encoding="utf-8")
                        father_node = data["father nodes"]

                        if experience_data[father_node]["score"] is None:
                            experience_data[father_node]["score"] = data["before"]

                        if data["succeed"]:
                            experience_data[father_node]["success"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                        else:
                            experience_data[father_node]["failure"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                except Exception as e:
                    logger.info(f"Error processing {round_path.name}: {str(e)}")

        experience_data = dict(experience_data)
        self.store.write_json(rounds_dir, "processed_experience.json", experience_data)
        logger.info(f"Processed experience data saved to {rounds_dir / 'processed_experience.json'}")
        return experience_data

    def format_experience(self, processed_experience, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            experience = f"Original Score: {experience_data['score']}\n"
            experience += "These are some conclusions drawn from experience:\n\n"
            for value in experience_data["failure"].values():
                experience += f"- Absolutely prohibit {value['modification']} (Score: {value['score']})\n"
            for value in experience_data["success"].values():
                experience += (
                    f"- Successful modification that can be referenced: "
                    f"{value['modification']} (Score: {value['score']})\n"
                )
            experience += (
                "\n\nNote: Take into account past failures and avoid repeating the same mistakes, "
                "as these failures indicate that these approaches are ineffective. You may refer to "
                "successful modifications as inspiration, but absolutely prohibit workflows that are "
                "too similar to previously successful ones. You must fundamentally change your way of "
                "thinking."
            )
        else:
            experience = f"No experience data found for round {sample_round}."
        return experience

    def check_modification(self, processed_experience, modification, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            for value in experience_data["failure"].values():
                if value["modification"] == modification:
                    return False
            for value in experience_data["success"].values():
                if value["modification"] == modification:
                    return False
            return True
        return True

    def create_experience_data(self, sample, modification):
        return {
            "father nodes": sample["round"],
            "modification": modification,
            "before": sample["score"],
            "after": None,
            "succeed": None,
        }

    def update_experience(self, directory, experience, avg_score):
        experience["after"] = avg_score
        experience["succeed"] = bool(avg_score > experience["before"])
        self.store.write_json(directory, "experience.json", experience)

    def _workflow_root(self, workflow_path: str | Path | None = None) -> Path:
        if workflow_path is not None:
            return Path(workflow_path)
        return self.store.workflow_path(self.workflow_name)
