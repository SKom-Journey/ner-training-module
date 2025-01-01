from typing import List, Optional, Dict

class MenuData:
    def __init__(self, source: str, title: str, description: str, image_url: str, ingredients: List[str], tags: List[str], comments: List[str], related_links: Optional[List[str]] = None):
        self.source = source
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
            "ingredients": self.ingredients,
            "comments": self.comments,
            
            # Exclude from json result
            # "imageUrl": self.image_url,
            # "related_links": self.related_links,
        }

        return data