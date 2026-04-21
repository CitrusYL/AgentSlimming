from src.evaluation.evaluator import Evaluator
from src.optimizer.workspace import WorkflowDirs


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(
        self,
        optimizer,
        graph_path,
        directory,
        validation_n,
        data,
        sample_size=None,
    ):
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for _ in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                sample_size=sample_size,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, sample_size=None, graph_path: str = None):
        evaluator = Evaluator(eval_path=directory)
        total_score = 0

        for _ in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                sample_size=sample_size,
            )

            cur_round = optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            target_graph_path = graph_path or str(optimizer.graph_utils.store.workflow_path(WorkflowDirs.GRAPH))
            result_path = optimizer.data_utils.get_results_file_path(target_graph_path)
            optimizer.data_utils.save_results(result_path, data)

            total_score += score

        return total_score / validation_n
