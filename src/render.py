import mitsuba as mi
import drjit as dr


def renderer(scene, params, spp=16):
    dr.sync_thread()
    image = mi.render(scene=scene, params=params, spp=spp)
    dr.sync_thread()
    return image