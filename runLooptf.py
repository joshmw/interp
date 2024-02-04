import subprocess
print('0118')

interps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
image_pairs = [
        ('in/grass.jpg', 'in/leaves.jpg'),
        ('in/pebbles2.jpg', 'in/granite.jpg'),
        ('in/b1.jpg', 'in/b6.jpg'),
        ('in/acorns.jpg', 'in/redwood.jpg'),
        ("in/canopy.jpg", "in/leaves.jpg"),
        ("in/canopy.jpg", "in/moss.jpg"),
        ("in/fern.jpg", "in/grass.jpg"),
        ("in/fern.jpg", "in/leaves.jpg"),
        ("in/lemons.jpg", "in/bananas.jpg"),
        ("in/mango.jpg", "in/yellowGems.jpg"),
        ("in/obsidian.jpg", "in/licorice.jpg"),
        ("in/ocean.jpg", "in/sky.jpg"),
        ("in/pine.jpg", "in/grass.jpg"),
        ("in/pine.jpg", "in/moss.jpg"),
        ("in/redwood.jpg", "in/rockwall.jpg"),
        ("in/rubies.jpg", "in/cherries.jpg"),
        ("in/snow.jpg", "in/pebbles2.jpg"),
        ("in/moss.jpg", "in/grass.jpg"),
        ("in/moss.jpg", "in/leaves.jpg")
]

for image1, image2 in image_pairs:
    for interp in interps:
        command = ["python", "synthesize.py", "-t", str(interp), "-k", "texture", "-i", image1, "-j", image2, "-o", "out/texture_pool4"]
        subprocess.run(command)

