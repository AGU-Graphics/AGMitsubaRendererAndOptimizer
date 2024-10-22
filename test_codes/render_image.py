import mitsuba as mi
import time
import numpy as np

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')

# Load the scene
scene = mi.load_file('scenes/true_scene_gray.xml')

start_time = time.time()

image = mi.render(scene)

# Save the image
mi.util.write_bitmap('data/true_image_gray.png', image, write_async=True)

# save the image to csv
np_image = np.array(image)
np.savetxt('data/true_image_gray.csv', np_image.reshape(-1, 3), delimiter=',', fmt='%f')

end_time = time.time()

print(f"Rendering time: {end_time - start_time} seconds")
