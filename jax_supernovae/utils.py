def integration_grid(low, high, target_spacing):
    """Divide the range between `low` and `high` into uniform bins
    with spacing less than or equal to `target_spacing` and return the
    bin midpoints and the actual spacing."""

    range_diff = high - low
    spacing = range_diff / int(math.ceil(range_diff / target_spacing))
    grid = np.arange(low + 0.5 * spacing, high, spacing)

    return grid, spacing 