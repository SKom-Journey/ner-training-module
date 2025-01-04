from typing import List, Optional, Dict

class MenuData:
    def __init__(self, source: str, title: str, description: str, image_url: str, nutritions: List[str], tips: List[str], steps: List[str], ingredients: List[str], tags: List[str], comments: List[str], related_links: Optional[List[str]] = None):
        self.steps = steps
        self.source = source
        self.nutritions = nutritions
        self.tips = tips
        self.title = title
        self.description = description
        self.image_url = image_url
        self.ingredients = ingredients
        self.comments = comments
        self.tags = tags
        self.related_links = related_links

    def to_dict(self) -> Dict:
        data = {
            "source": self.source,
            "tags": self.tags,
            "title": self.title,
            "description": self.description,
            
            # Exclude from json result
            # "comments": self.comments,
            # "ingredients": self.ingredients,
            # "imageUrl": self.image_url,
            # "related_links": self.related_links,
        }

        return data
    
    def to_dataset(self) -> List[str]:
        data = [
            self.title,
            self.description,
           *self.ingredients,
           *self.steps,
           *self.tips,
           *self.comments,
           *self.nutritions,
           *self.tags
        ]

        return data