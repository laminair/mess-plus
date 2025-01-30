

def is_nested_list(obj):

    return isinstance(obj, list) and any(isinstance(item, list) for item in obj)