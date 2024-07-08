"""
Just a regular function wrapper to transform text to tokens
    args: (transform: list of functions)
    Return the combination of all the transform functions
"""
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func