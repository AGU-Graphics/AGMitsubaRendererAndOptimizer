import mitsuba as mi


def renderer(scene, params, spp=16):
    return mi.render(scene=scene, params=params, spp=spp)