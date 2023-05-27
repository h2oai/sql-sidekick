import json
from typing import Dict, List


# Reference: https://python.langchain.com/en/latest/modules/memory/examples/custom_memory.html
class EntityMemory:
    def __init__(self, k, path: str = None):
        self.k = k
        self.track_history: List = []
        self.track_entity: List = []
        self.path = path

    def extract_entity():
        # Currently, anything tagged between below tags are extracted respectively,
        # 1. From Input text: <context_key> some key </context_key>
        # 2. From Output text: <context_value> some key </context_value>
        {
            "CPU": "payload->'engineEvent'-> 'pausing' -> 'engine'->> 'cpu'",
            "global usage DAI instance": "GREATEST((gpu * 4.0)) as ai_units",
            "global usage": "GREATEST((GREATEST((ram_gi / 64.0), (cpu / 8.0)) - gpu), 0) + (gpu * 4.0) as ai_units",
        }

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
        if query.strip() and "SELECT".lower() in response.lower():
            chat_history = {"Query": query, "Answer": response}
            self.track_history.append(chat_history)
        else:
            raise ValueError("Response not valid. Please try again.")

        # Add logic for entity extraction
        if extract_context:
            # Generic logic pending
            extracted_entity = {"GPU": "payload->'engineEvent'-> 'pausing' -> 'engine'->> 'gpu'"}
            self.track_entity.append(extracted_entity)

        # persist the information for future use
        res = {"history": self.track_history, "entity": self.track_entity}

        with open(f"{self.path}/var/lib/tmp/data/history.jsonl", "a") as outfile:
            for entry in self.track_history:
                json.dump(entry, outfile)
                outfile.write("\n")
        return res
