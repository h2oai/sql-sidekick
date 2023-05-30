import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# Reference: https://python.langchain.com/en/latest/modules/memory/examples/custom_memory.html
class EntityMemory:
    def __init__(self, k, path: str = None):
        self.k = k
        self.track_history: List = []
        self.track_entity: List = []
        self.path = path

    def extract_entity(self, question: str, answer: str) -> Tuple[List, List]:
        # Currently, anything tagged between below tags are extracted respectively,
        # 1. From Input text: <context_key> some key </context_key>
        # 2. From Output text: <context_value> some key </context_value>
        # TODO Chat mode for auto extraction of entities
        c_k = re.findall(r"<key>(.+?)</key>", question)
        c_val = re.findall(r"<value>(.+?)</value>", answer)
        return (c_k, c_val)

    def save_context(self, info: str, extract_context: bool = True) -> Dict:
        # Construct dictionary to record history
        # {
        # 'Query':
        # 'Answer':
        # }
        # Extract info from the supplied text
        split_token = ";"
        query = " ".join(info.partition(":")[2].split(split_token)[0].strip().split())
        response = " ".join(info.partition(":")[2].split(split_token)[1].partition(":")[2].strip().split())
        # TODO add additional guardrails to check if the response is a valid response.
        # At-least syntactically correct SQL.

        # Check if entity extraction is enabled
        # Add logic for entity extraction
        extracted_entity = None
        if extract_context:
            _k, _v = self.extract_entity(query, response)
            k_v = " ".join(_k)
            c_v = ", ".join(_v)
            extracted_entity = {k_v: c_v}
            self.track_entity.append(extracted_entity)

        chat_history = {}
        if query.strip() and "SELECT".lower() in response.lower():
            chat_history = {"Query": query, "Answer": response, "entity": extracted_entity}
            self.track_history.append(chat_history)
        else:
            raise ValueError("Response not valid. Please try again.")
        # persist the information for future use
        res = {"history": self.track_history, "entity": self.track_entity}

        # Persist added information locally
        if chat_history:
            with open(f"{self.path}/var/lib/tmp/data/history.jsonl", "a") as outfile:
                json.dump(chat_history, outfile)
                outfile.write("\n")
            if extract_context:
                # Update context.json file for tracking entities
                content_file_path = f"{self.path}/var/lib/tmp/data/context.json"
                context_dict = extracted_entity
                if Path(content_file_path).exists():
                    context_dict = json.load(open(content_file_path, "r"))
                    context_dict.update(extracted_entity)
                with open(content_file_path, "w") as outfile:
                    json.dump(context_dict, outfile, indent=4, sort_keys=False)
        return res
