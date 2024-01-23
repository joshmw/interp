import subprocess
print('0118')

interps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
image_pairs = [('in/grass.jpg', 'in/leaves.jpg'),
               ('in/pebbles2.jpg', 'in/granite.jpg'),
               ('in/b1.jpg', 'in/b6.jpg')]

for image1, image2 in image_pairs:
    for interp in interps:
        command = ["python3", "synthesize.py", "-t", str(interp), "-k", "activation", "-i", image1, "-j", image2]
        subprocess.run(command)

