"""Small, explicit fixtures for offline examples (not learned predictions)."""


def tiny_scene():
    return {
        "image_index": 0,
        "objects": [
            {"color": "red", "shape": "cube", "size": "small", "material": "rubber"},
            {"color": "blue", "shape": "sphere", "size": "large", "material": "metal"},
        ],
        "relationships": {
            "left": [[], [0]], "right": [[1], []],
            "front": [[], []], "behind": [[], []],
        },
    }


def tiny_question(attribute="color"):
    """What attribute has the sphere to the right of the red cube?"""
    steps = [
        ("scene", []), ("filter_color", ["red"]), ("filter_shape", ["cube"]),
        ("unique", []), ("relate", ["right"]), ("filter_shape", ["sphere"]),
        ("unique", []), (f"query_{attribute}", []),
    ]
    return {"question": f"What {attribute} is the sphere right of the red cube?",
            "program": [{"function": name, "value_inputs": values,
                         "inputs": [] if i == 0 else [i - 1]}
                        for i, (name, values) in enumerate(steps)]}
