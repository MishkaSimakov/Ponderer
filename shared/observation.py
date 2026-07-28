COLUMNS = ["distance", "left_color", "right_color", "left_position", "right_position"]

DIM = len(COLUMNS)


def shape(raw):
    """Raw sensors to policy input. Passthrough until there is a policy to feed."""
    return list(raw)
