from ..constants import ASSISTANT, PROMPT_PREFIX_DICT, USER


class PromptTemplate:
    def __init__(self, prompt_prefix: str = "", with_incontext: bool = False) -> None:
        self.prompt_prefix = prompt_prefix
        self.with_incontext = with_incontext

    def get(self, data: dict) -> str:
        '''Format the instruction as a prompt for LLM.'''
        question_and_choice = f"{data['question']} \nA.{data['A']} \nB.{data['B']} \nC.{data['C']} \nD.{data['D']}".replace(
            " ", "")

        if self.with_incontext:
            return f"""{PROMPT_PREFIX_DICT.get(self.prompt_prefix, "")}你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。提供你例子參考: USER: 多數國家的民族分布，存在「大分散，小聚居」的現象，如果政府欲增進不同民族間的關係和諧，下列何種作法最為適當？\nA. 經由政黨協商，自行劃定各民族的自治區\nB. 鼓勵各民族的交流，理性對待彼此的文化差異\nC. 開放不同民族間的競爭，創造最優秀的強勢民族\nD. 將相同語言、文化的民族遷徙集中到同一生活棲息地 ASSISTANT: B. 鼓勵各民族的交流，理性對待彼此的文化差異。因為國父民族主義主張民族平等，尤其複合民族國家間，要增進不同民族間之和諧，需堅持民族平等的原則，並強調文化多元主義的觀點，選項(B)，即符合此一文化多元化之理想；至於選項A、C及D，或剝奪弱勢族群的基本權利，或要求弱勢族群放棄自己的文化，接受優勢族群的文化，或多或少都具有種族中心主義的色彩，不符合民族主義所強調之文化相對論原則。現在請回答: {USER}: {question_and_choice} {ASSISTANT}: 正確答案為"""
        else:
            return f"""{PROMPT_PREFIX_DICT.get(self.prompt_prefix, "")}你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。{USER}: {question_and_choice} {ASSISTANT}: 正確答案為"""


class Answer:
    def __init__(self, with_answer_details: bool = False) -> None:
        self.with_answer_details = with_answer_details

    def get(self, data: dict) -> str:
        '''Format the answer as a prompt for LLM.'''
        if self.with_answer_details:
            return f"{data['answer']}.{data[str(data['answer'])]} 原因：{data['answer_details']}"
        else:
            # Example: "A. XXXX"
            return f"{data['answer']}.{data[str(data['answer'])]}"
