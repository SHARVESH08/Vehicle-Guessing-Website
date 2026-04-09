import os
import matplotlib.pyplot as plt

root = r"C:\VIT Stuff\SEM-4\FoDS\Project\Dataset\train"

class_counts = {}

for cls in os.listdir(root):
    class_counts[cls] = len(os.listdir(os.path.join(root, cls)))

counts = list(class_counts.values())

print("Total Classes:", len(class_counts))
print("Total Images:", sum(counts))

plt.figure(figsize=(8,5))
plt.hist(counts, bins=20)
plt.title("Distribution of Images per Class")
plt.xlabel("Number of Images")
plt.ylabel("Number of Classes")
plt.show()
