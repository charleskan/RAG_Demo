from typing import Any, Dict, List

class Document:
    """
    Represents a document with user information, file information, document embedding, and details.
    """

    def __init__(self, 
                 metadata: dict[str, any],
                 content: str):
        """
        Initializes a new document object.
        
        :param metadata: A dictionary containing metadata about the document.
        :param context: The text content of the document.
        """
        self.metadata = metadata
        self.content = content


