from pydantic import BaseModel, Field


class CustomResponseModel(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class ProgramResponseModel(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class AnswerGenerateResponseModel(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class ScEnsembleResponseModel(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class TestResponseModel(BaseModel):
    reflection: str = Field(default="", description="Your reflection on the test cases and code solution.")
    solution: str = Field(default="", description="Your improved code solution.")
