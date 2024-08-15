import subprocess
print('0815')

interps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#interps = [0.0, 1.0]
#interps = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
image_pairs = [
    ('in/acorns.jpg', 'in/redwood.jpg'),
    #('in/b1.jpg', 'in/b6.jpg'),
 	#('in/canopy.jpg', 'in/moss.jpg'),
	#('in/fern.jpg', 'in/grass.jpg'),
	#('in/fern.jpg', 'in/leaves.jpg'),
	('in/grass.jpg', 'in/leaves.jpg'),
	('in/lemons.jpg', 'in/bananas.jpg'),
#	('in/mango.jpg', 'in/yellowGems.jpg'),
	#('in/obsidian.jpg', 'in/licorice.jpg'),
	#('in/ocean.jpg', 'in/sky.jpg'),
	#('in/pebbles.jpg', 'in/granite.jpg'),
	#('in/pine.jpg', 'in/grass.jpg'),
	#('in/pine2.jpg', 'in/moss.jpg'),
	#('in/redwood.jpg', 'in/rockwall.jpg'),
	#('in/rubies.jpg', 'in/cherries.jpg'),
	#('in/moss2.jpg', 'in/grass.jpg'),
	#('in/moss3.jpg', 'in/leaves.jpg'),
	#('in/blueberries.jpg', 'in/beads.jpg'),
	#('in/autumn.jpg', 'in/fire.jpg'),
	('in/petals.jpg', 'in/buttercream.jpg'),
	#('in/brick.jpg', 'in/redwood.jpg'),
	#('in/orangePeel.jpg', 'in/orangeFabric.jpg'),
     #('in/lemons.jpg', 'in/grass.jpg'),
     #('in/lemons.jpg', 'in/leaves.jpg'),
     #('in/bananas.jpg', 'in/grass.jpg'),
     #('in/bananas.jpg', 'in/leaves.jpg'),
]

for image1, image2 in image_pairs:
    for interp in interps:
        command = ["python", "synthesize.py", "-t", str(interp), "-p", str(10), "-i", image1, "-j", image2, "-s", str(5)]
        subprocess.run(command)
