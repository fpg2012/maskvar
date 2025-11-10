class InstanceInfo:
    """
    Encapsulates hierarchical information about a dataset instance.
    
    Attributes:
        mapping (tuple): (layer_index, mask_id) in layers tensor
        parent (int or None): Parent instance ID
        children (list): List of child instance IDs
        node_level (int): Depth in hierarchy
    """
    def __init__(self, mapping=None, parent=None, children=None, node_level=0):
        self.mapping = mapping
        self.parent = parent
        self.children = children if children is not None else []
        self.node_level = node_level

    def to_dict(self):
        """Converts the instance info to a dictionary."""
        return {
            'mapping': self.mapping,
            'parent': self.parent,
            'children': self.children,
            'node_level': self.node_level
        }

    @classmethod
    def from_dict(cls, data):
        """Creates an instance from a dictionary."""
        return cls(
            mapping=data.get('mapping'),
            parent=data.get('parent'),
            children=data.get('children', []),
            node_level=data.get('node_level', 0)
        )
    
    def __str__(self) -> str:
        return f"InstanceInfo(mapping={self.mapping}, parent={self.parent}, children={self.children}, node_level={self.node_level})"
    
    def __repr__(self) -> str:
        return f"InstanceInfo(mapping={self.mapping}, parent={self.parent}, children={self.children}, node_level={self.node_level})"