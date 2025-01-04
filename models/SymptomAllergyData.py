from typing import List, Dict

class SymptomAllergyData:
    def __init__(self, source: str, title: str, paragraphs: List[str], headings: List[str]):
        self.source = source
        self.title = title
        self.paragraphs = paragraphs
        self.headings = headings

    def to_dict(self) -> Dict:
        data = {
            "source": self.source,
            "title": self.title,
            "paragraphs": self.paragraphs,
            "headings": self.headings,
            
            # Exclude from json result
        }

        return data
    
    def to_dataset(self) -> List[str]:
        data = [
            self.title,
            *self.paragraphs,
            *self.headings,
        ]

        return data