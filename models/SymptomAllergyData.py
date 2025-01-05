from typing import List, Dict

class SymptomAllergyData:
    def __init__(self, source: str, title: str, paragraphs: List[str], headings: List[str], bullets: List[str]):
        self.source = source
        self.title = title
        self.paragraphs = paragraphs
        self.headings = headings
        self.bullets = bullets

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
            *self.bullets,
            *self.headings,
        ]

        return data