def get_patch_coordinates(slide, patch_size=512, level=0, stride=None):
    if stride is None:
        stride = patch_size
    width, height = slide.level_dimensions[level]
    return [
        (x, y)
        for y in range(0, height - patch_size + 1, stride)
        for x in range(0, width - patch_size + 1, stride)
    ]
